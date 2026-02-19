import copy
import torch
from torch import nn as nn
from torch.nn import functional as F
#from CASA-NER.modeling_albert import AlbertModel, AlbertMLMHead
from CASA-NER.modeling_bert import BertConfig, BertModel
#from CASA-NER.modeling_roberta import RobertaConfig, RobertaModel, RobertaLMHead
#from CASA-NER.modeling_xlm_roberta import XLMRobertaConfig
from transformers.modeling_utils import PreTrainedModel
from CASA-NER.SMAM import Semantic Modulation Attention
from CASA-NER import util
import logging

logger = logging.getLogger()

# #Context-Aware Feature Enhancement (ASRN) Module
class ContextAwareEntityEnhancement(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.context_attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout)
        self.context_encoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, 1)  # Gate to control the contribution of enhanced features
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embeddings, entity_embeddings, token_mask):
        # 使用自注意力机制捕捉上下文信息
        context_output, _ = self.context_attention(token_embeddings, token_embeddings, token_embeddings)
        # 将上下文增强后的实体嵌入与原始实体嵌入结合
        enhanced_entity_embeddings = self.fc(entity_embeddings + context_output)
        enhanced_entity_embeddings = self.dropout(enhanced_entity_embeddings)

        # 自适应选择融合原始嵌入和增强嵌入
        gate = torch.sigmoid(self.gate(enhanced_entity_embeddings))
        final_entity_embeddings = gate * enhanced_entity_embeddings + (1 - gate) * entity_embeddings
        return final_entity_embeddings

#Complexity-Aware Strategy Selection Generator (CASSG) Module
class DynamicPromptGenerator(nn.Module):
   
    def __init__(
        self,
        hidden_size: int,
        prompt_length: int,
        n_prompts=(20, 35, 50),
        top_c: int = 2,
        shallow_layers: int = 8,
        deep_layers: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.prompt_length = prompt_length
        self.n1, self.n2, self.n3 = n_prompts
        self.top_c = top_c
        self.shallow_layers = shallow_layers
        self.deep_layers = deep_layers


        # lexical + syntactic + semantic -> fused feature vector z
        self.fuse = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )


        self.prompt_bank_1 = nn.Parameter(torch.randn(self.n1, hidden_size * 2) * 0.02)
        self.prompt_bank_2 = nn.Parameter(torch.randn(self.n2, hidden_size * 2) * 0.02)
        self.prompt_bank_3 = nn.Parameter(torch.randn(self.n3, hidden_size * 2) * 0.02)

        self.strategy_mod_1 = nn.Linear(hidden_size, hidden_size * 2)
        self.strategy_mod_2 = nn.Linear(hidden_size, hidden_size * 2)
        self.strategy_mod_3 = nn.Linear(hidden_size, hidden_size * 2)

        self.strategy_scorer = nn.Linear(hidden_size, 3)


        self.activation_scale = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )


        self.position_proj = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.type_proj = nn.Linear(hidden_size * 2, hidden_size * 2)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int):
        mask_f = mask.float()
        denom = mask_f.sum(dim=dim, keepdim=True).clamp_min(1.0)
        return (x * mask_f.unsqueeze(-1)).sum(dim=dim, keepdim=True) / denom.unsqueeze(-1)

    def _complexity_signals(self, hidden_states, attention_mask: torch.Tensor):
      
        eps = 1e-6
        last = hidden_states[-1]  # (B, L, H)
        mask = attention_mask.bool()

        # Lexical
        token_var = last.var(dim=-1, unbiased=False)  # (B, L)
        lexical = (token_var * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1.0)  # (B,)

        # Syntactic
        n_layers = len(hidden_states)
        shallow_n = min(self.shallow_layers, n_layers)
        deep_n = min(self.deep_layers, n_layers)

        shallow_stack = torch.stack(hidden_states[:shallow_n], dim=0).mean(dim=0)  # (B, L, H)
        deep_stack = torch.stack(hidden_states[-deep_n:], dim=0).mean(dim=0)       # (B, L, H)

        shallow_sent = self._masked_mean(shallow_stack, mask, dim=1).squeeze(1)  # (B, H)
        deep_sent = self._masked_mean(deep_stack, mask, dim=1).squeeze(1)        # (B, H)

        syn_cos = F.cosine_similarity(shallow_sent, deep_sent, dim=-1).clamp(-1+eps, 1-eps)
        syntactic = 1.0 - syn_cos

        # Semantic
        x1 = last[:, :-1, :]
        x2 = last[:, 1:, :]
        m_pair = mask[:, :-1] & mask[:, 1:]
        pair_cos = F.cosine_similarity(x1, x2, dim=-1)  # (B, L-1)
        semantic = 1.0 - ((pair_cos * m_pair.float()).sum(dim=1) / m_pair.float().sum(dim=1).clamp_min(1.0))

        return torch.stack([lexical, syntactic, semantic], dim=-1)  # (B, 3)

    def _build_candidates(self, z: torch.Tensor):
        B = z.size(0)
        d1 = self.strategy_mod_1(z).unsqueeze(1)
        d2 = self.strategy_mod_2(z).unsqueeze(1)
        d3 = self.strategy_mod_3(z).unsqueeze(1)

        p1 = self.prompt_bank_1.unsqueeze(0).expand(B, -1, -1) + d1
        p2 = self.prompt_bank_2.unsqueeze(0).expand(B, -1, -1) + d2
        p3 = self.prompt_bank_3.unsqueeze(0).expand(B, -1, -1) + d3
        return self.dropout(p1), self.dropout(p2), self.dropout(p3)

    def _select_and_merge(self, candidates, weights: torch.Tensor):
        p1, p2, p3 = candidates
        B = weights.size(0)

        topv, topi = torch.topk(weights, k=min(self.top_c, 3), dim=-1)  # (B, C)
        pools = []
        sizes = []
        for b in range(B):
            parts = []
            for r in range(topi.size(1)):
                sid = int(topi[b, r].item())
                w = topv[b, r].item()
                if sid == 0:
                    parts.append(p1[b] * w)
                elif sid == 1:
                    parts.append(p2[b] * w)
                else:
                    parts.append(p3[b] * w)
            pool = torch.cat(parts, dim=0)
            pools.append(pool)
            sizes.append(pool.size(0))

        max_pool = max(sizes)
        pool_tensor = p1.new_zeros((B, max_pool, p1.size(-1)))
        pool_mask = torch.zeros((B, max_pool), dtype=torch.bool, device=p1.device)
        for b, pool in enumerate(pools):
            n = pool.size(0)
            pool_tensor[b, :n] = pool
            pool_mask[b, :n] = True
        return pool_tensor, pool_mask

    def _activate_prompts(self, pool: torch.Tensor, pool_mask: torch.Tensor, z: torch.Tensor, text_repr: torch.Tensor):
        """
        Adaptive prompt activation (Eq.9-11).
        Outputs variable-length prompts per batch; downstream can pad/truncate as needed.
        """
        B, Npool, D = pool.shape
        H = self.hidden_size

        scale = torch.sigmoid(self.activation_scale(z)).squeeze(-1)  # (B,)
        expected = scale * (self.n1 + self.n2 + self.n3) / 3.0
        n_active = expected.round().long().clamp(min=1, max=Npool)

        prompt_repr = pool[:, :, :H]
        scores = torch.einsum("bnh,bh->bn", prompt_repr, text_repr)
        scores[~pool_mask] = -1e9

        max_k = int(n_active.max().item())
        out = pool.new_zeros((B, max_k, D))
        out_mask = torch.zeros((B, max_k), dtype=torch.bool, device=pool.device)

        for b in range(B):
            k = int(n_active[b].item())
            idx = torch.topk(scores[b], k=min(k, scores.size(1)), dim=-1).indices
            sel = pool[b, idx]
            out[b, :k] = sel[:k]
            out_mask[b, :k] = True

        return out, out_mask

    def forward(self, hidden_states, attention_mask: torch.Tensor):
        signals = self._complexity_signals(hidden_states, attention_mask)
        z = self.fuse(signals)

        last = hidden_states[-1]
        text_repr = self._masked_mean(last, attention_mask.bool(), dim=1).squeeze(1)

        candidates = self._build_candidates(z)
        weights = F.softmax(self.strategy_scorer(z), dim=-1)

        pool, pool_mask = self._select_and_merge(candidates, weights)
        prompts, prompt_mask = self._activate_prompts(pool, pool_mask, z, text_repr)

        position_queries = self.position_proj(prompts)
        type_queries = self.type_proj(prompts)
        return position_queries, type_queries, prompt_mask

    def generate_dynamic_prompts(self, context_embeddings, context_masks):
     
        batch_size = context_embeddings.shape[0]
   
        dynamic_prompts = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        return dynamic_prompts

    def forward(self, context_embeddings, context_masks):
        dynamic_prompts = self.generate_dynamic_prompts(context_embeddings, context_masks)
        return dynamic_prompts


class EntityBoundaryPredictor(nn.Module):
    def __init__(self, config, prop_drop):
        super().__init__()
        self.config = config
        self.prop_drop = prop_drop
        self.hidden_size = config.hidden_size
        self.token_embedding_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size)
        ) 
        self.entity_embedding_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size)
        ) 
        self.boundary_predictor = nn.Linear(self.hidden_size, 1)
    
    def forward(self, token_embedding, entity_embedding, token_mask):
        # B x #ent x #token x hidden_size
        entity_token_matrix = self.token_embedding_linear(token_embedding).unsqueeze(1) + self.entity_embedding_linear(entity_embedding).unsqueeze(2)
        entity_token_cls = self.boundary_predictor(torch.tanh(entity_token_matrix)).squeeze(-1)
        token_mask = token_mask.unsqueeze(1).expand(-1, entity_token_cls.size(1), -1)
        entity_token_cls[~token_mask] = -1e25
        # entity_token_p = entity_token_cls.softmax(dim=-1)
        entity_token_p = F.sigmoid(entity_token_cls)
        return entity_token_p

class EntityBoundaryPredictorBak(nn.Module):
    def __init__(self, config, prop_drop):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.token_embedding_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(prop_drop)
        ) 
        self.entity_embedding_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(prop_drop)
        ) 
        self.boundary_predictor = nn.Linear(self.hidden_size, 1)
    
    def forward(self, token_embedding, entity_embedding, token_mask):
        entity_token_matrix = self.token_embedding_linear(token_embedding).unsqueeze(1) + self.entity_embedding_linear(entity_embedding).unsqueeze(2)
        entity_token_cls = self.boundary_predictor(torch.relu(entity_token_matrix)).squeeze(-1)
        token_mask = token_mask.unsqueeze(1).expand(-1, entity_token_cls.size(1), -1)
        entity_token_cls[~token_mask] = -1e30
        entity_token_p = F.sigmoid(entity_token_cls)

        return entity_token_p


class EntityTypePredictor(nn.Module):
    def __init__(self, config, entity_type_count, mlm_head):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, entity_type_count),
        )
    
    def forward(self, h_cls):
        entity_logits = self.classifier(h_cls)
        return entity_logits

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class DetrTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=768, d_ffn=1024, dropout=0.1, activation="relu", n_heads=8, selfattn = True, ffn = True):
        super().__init__()

        # cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.selfattn = selfattn
        self.ffn = ffn


        if selfattn:
            # self attention
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(d_model)
        if ffn:
            # ffn
            self.linear1 = nn.Linear(d_model, d_ffn)
            self.activation = _get_activation_fn(activation)
            self.dropout3 = nn.Dropout(dropout)
            self.linear2 = nn.Linear(d_ffn, d_model)
            self.dropout4 = nn.Dropout(dropout)
            self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, pos, src, mask):
        if self.selfattn:
            q = k = self.with_pos_embed(tgt, pos)
            v = tgt
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))[0].transpose(0, 1)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        q = self.with_pos_embed(tgt, pos)
        k = v = src
        tgt2 = self.cross_attn(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), key_padding_mask=~mask if mask is not None else None)[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if self.ffn:
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout4(tgt2)
            tgt = self.norm3(tgt)

        return tgt

class DetrTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, pos, src, mask):
        output = tgt

        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            output = layer(output, tgt, src, mask)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output

class promptner(PreTrainedModel):

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _compute_extended_attention_mask(self, attention_mask, context_count, prompt_number):
        
        if not self.prompt_individual_attention and not self.sentence_individual_attention:
            # #batch x seq_len
            extended_attention_mask = attention_mask
        else:
            # #batch x seq_len x seq_len
            extended_attention_mask = attention_mask.unsqueeze(1).expand(-1, attention_mask.size(-1), -1).clone()

            for mask, c_count in zip(extended_attention_mask, context_count):
                # mask seq_len x seq_len
                # mask prompt for sentence encoding
                if self.prompt_individual_attention:
                    # encode for each prompt
                    for p in range(prompt_number):
                        mask[p*self.prompt_length:  p*self.prompt_length + self.prompt_length, :prompt_number*self.prompt_length] = 0
                        mask[p*self.prompt_length: p*self.prompt_length + self.prompt_length, p*self.prompt_length: p*self.prompt_length + self.prompt_length] = 1
                if self.sentence_individual_attention:
                    for c in range(c_count):
                        mask[c+self.prompt_length*prompt_number, :self.prompt_length*prompt_number] = 0

        return extended_attention_mask

    def __init__(
        self,
        model_type, 
        config, 
        entity_type_count: int, 
        prop_drop: float, 
        freeze_transformer: bool, 
        lstm_layers = 3, 
        decoder_layers = 3,
        pool_type:str = "max", 
        prompt_individual_attention = True, 
        sentence_individual_attention = True,
        use_masked_lm = False, 
        last_layer_for_loss = 3, 
        split_epoch = 0, 
        clip_v = None,
        prompt_length = 3,
        prompt_number = 60,
        prompt_token_ids = None):
        super().__init__(config)
        
 
        self.freeze_transformer = freeze_transformer
        self.split_epoch = split_epoch
        self.has_changed = False
        self.loss_layers = last_layer_for_loss
        self.model_type = model_type
        self.use_masked_lm = use_masked_lm
        self._entity_type_count = entity_type_count
        self.prop_drop = prop_drop
        self.split_epoch = split_epoch
        self.lstm_layers = lstm_layers
        self.prompt_individual_attention = prompt_individual_attention
        self.sentence_individual_attention = sentence_individual_attention


        self.decoder_layers = decoder_layers
        self.prompt_number = prompt_number
        self.prompt_length = prompt_length
        self.pool_type = pool_type

        
        # AddComplexity-Aware Strategy Selection Generator (CASSG) Module
        self.dynamic_prompt_generator = DynamicPromptGenerator(config.hidden_size, prompt_length,
            n_prompts=(20, 35, 50), top_c=2, dropout=0.1)

        #初始化
        self.SMAM_attention = Semantic Modulation Attention(
            dim=config.hidden_size,  # 对应嵌入维度
            num_heads=8,            # 设置多头数量（可调整）
            qkv_bias=True,          # 是否添加偏置
            qk_norm=False,          # 是否归一化
            attn_drop=0.1,          # 注意力层 dropout
            proj_drop=0.1,          # 输出层 dropout
            shared_head=4,          # 设置共享头的数量
            routed_head=4,          # 设置路由头的数量
            head_dim=64             # 每头的维度（可调整）
        )

        self.withimage = False
        if clip_v is not None:
            self.withimage = True
        
        self.query_embed = nn.Embedding(prompt_number, config.hidden_size * 2)

        if self.decoder_layers > 0:
            decoder_layer = DetrTransformerDecoderLayer(d_model=config.hidden_size, d_ffn=1024, dropout=0.1, selfattn = True, ffn = True)
            self.decoder = DetrTransformerDecoder(decoder_layer=decoder_layer, num_layers=self.decoder_layers)
            if self.withimage:
                self.img_decoder = DetrTransformerDecoder(decoder_layer=decoder_layer, num_layers=self.decoder_layers)
            if self.prompt_length>1:
                self.decoder2 = DetrTransformerDecoder(decoder_layer=decoder_layer, num_layers=self.decoder_layers)

        # #Add Context-Aware Entity Enhancement (ASRN)
        self.context_aware_enhancer = ContextAwareEntityEnhancement(config.hidden_size, dropout=0.1)


        if model_type == "roberta":
            self.roberta = RobertaModel(config)
            self.model = self.roberta
            self.lm_head = RobertaLMHead(config)
            self.entity_classifier = EntityTypePredictor(config, entity_type_count, lambda x: self.lm_head(x))

        if model_type == "bert":
            # self.bert = BertModel(config)
            self.prompt_ids = prompt_token_ids
            self.bert = BertModel(config, prompt_ids = prompt_token_ids)
            self.model = self.bert
            for name, param in self.bert.named_parameters():
                if "pooler" in name:
                    param.requires_grad = False
            # self.cls = BertOnlyMLMHead(config)
            self.cls = None
            self.entity_classifier = EntityTypePredictor(config, entity_type_count, lambda x: self.cls(x))

        if model_type == "albert":
            # self.bert = BertModel(config)
            self.prompt_ids = prompt_token_ids
            self.bert = AlbertModel(config)
            self.model = self.bert
            self.predictions = AlbertMLMHead(config)
            self.entity_classifier = EntityTypePredictor(config, entity_type_count, lambda x: self.predictions(x))

        if self.withimage:
            self.vision2text = nn.Linear(clip_v.config.hidden_size, config.hidden_size)

        promptner._keys_to_ignore_on_save = ["model." + k for k,v in self.model.named_parameters()]
        # promptner._keys_to_ignore_on_load_unexpected = ["model." + k for k,v in self.model.named_parameters()]
        promptner._keys_to_ignore_on_load_missing = ["model." + k for k,v in self.model.named_parameters()]

        if self.lstm_layers > 0:
            self.lstm = nn.LSTM(input_size = config.hidden_size, hidden_size = config.hidden_size//2, num_layers = lstm_layers,  bidirectional = True, dropout = 0.1, batch_first = True)

        self.left_boundary_classfier = EntityBoundaryPredictor(config, self.prop_drop)
        self.right_boundary_classfier = EntityBoundaryPredictor(config, self.prop_drop)
        # self.entity_classifier = EntityTypePredictor(config, config.hidden_size, entity_type_count)
        self.init_weights()

        self.clip_v = clip_v

        if freeze_transformer or self.split_epoch > 0:
            logger.info("Freeze transformer weights")
            if self.model_type == "bert":
                model = self.bert
                mlm_head = self.cls
            if self.model_type == "roberta":
                model = self.roberta
                mlm_head = self.lm_head
            if self.model_type == "albert":
                model = self.albert
                mlm_head = self.predictions
            for name, param in model.named_parameters():
                param.requires_grad = False

    def _common_forward(
        self, 
        encodings: torch.tensor, 
        context_masks: torch.tensor, 
        raw_context_masks: torch.tensor, 
        inx4locator: torch.tensor, 
        pos_encoding: torch.tensor, 
        seg_encoding: torch.tensor, 
        context2token_masks:torch.tensor,
        token_masks:torch.tensor,
        image_inputs: dict = None,
        meta_doc = None):
        
        batch_size = encodings.shape[0]
        context_masks = context_masks.float()
        token_count = token_masks.long().sum(-1,keepdim=True)
        context_count = context_masks.long().sum(-1,keepdim=True)
        raw_context_count = raw_context_masks.long().sum(-1,keepdim=True)
        pos = None
        tgt = None
        tgt2 = None

        # pdb.set_trace()
        
        context_masks = self._compute_extended_attention_mask(context_masks, raw_context_count, self.prompt_number)
        # self = self.eval()
        if self.model_type == "bert":
            model = self.bert
        if self.model_type == "roberta":
            model = self.roberta
        # model.embeddings.position_embeddings
        outputs = model(
                    input_ids=encodings,
                    attention_mask=context_masks,
                    # token_type_ids=seg_encoding,
                    # position_ids=pos_encoding,
                    output_hidden_states=True)
        # last_hidden_state, pooler_output, hidden_states 
        
        
        # Generate dynamic prompts (CASSG): position / type queries for decoding
        position_queries, type_queries, prompt_mask = self.dynamic_prompt_generator(outputs.hidden_states, raw_context_masks)

      
        pos_q, tgt_q = torch.split(position_queries, outputs.last_hidden_state.size(2), dim=-1)
        pos_t, tgt_t = torch.split(type_queries, outputs.last_hidden_state.size(2), dim=-1)

        # Pad/Truncate to fixed prompt_number required by decoder implementation
        K = self.prompt_number  # fixed length used by decoders

        def _pad_to_k(x, k):
            n = x.size(1)
            if n == k:
                return x
            if n > k:
                return x[:, :k, :]
            pad = x.new_zeros((x.size(0), k - n, x.size(2)))
            return torch.cat([x, pad], dim=1)

        def _pad_mask(m, k):
            n = m.size(1)
            if n == k:
                return m
            if n > k:
                return m[:, :k]
            pad = torch.zeros((m.size(0), k - n), dtype=torch.bool, device=m.device)
            return torch.cat([m, pad], dim=1)

        pos_q = _pad_to_k(pos_q, K)
        tgt_q = _pad_to_k(tgt_q, K)
        pos_t = _pad_to_k(pos_t, K)
        tgt_t = _pad_to_k(tgt_t, K)
        prompt_mask = _pad_mask(prompt_mask, K)


        tgt = tgt_q
        pos = pos_q
        tgt2 = tgt_t
        pos2 = pos_t

        orig_tgt = tgt
        orig_tgt2 = tgt2

        intermediate = []
        for i in range(self.loss_layers, 0, -1):

            h = outputs.hidden_states[-1]
            h_token = util.combine(h, context2token_masks, self.pool_type)

            

            h_token = self.SMAM_attention(h_token)
    
            

            if self.lstm_layers > 0:
                h_token = nn.utils.rnn.pack_padded_sequence(input = h_token, lengths = token_count.squeeze(-1).cpu().tolist(), enforce_sorted = False, batch_first = True)
                h_token, (_, _) = self.lstm(h_token)
                h_token, _ = nn.utils.rnn.pad_packed_sequence(h_token, batch_first=True)

            if inx4locator is not None:
                
                tgt = util.batch_index(outputs.hidden_states[-i], inx4locator) + orig_tgt
                if self.prompt_length > 1:
                    tgt2 = util.batch_index(outputs.hidden_states[-i], inx4locator + self.prompt_length-1) + orig_tgt2

            updated_tgt = tgt

            if tgt2 is None:
                updated_tgt2 = tgt
            else:
                updated_tgt2 = tgt2

            if self.decoder_layers > 0:
                if self.withimage:
                    tgt = self.img_decoder(tgt, pos, aligned_image_h, mask=None)
                updated_tgt = self.decoder(tgt, pos, h_token, mask=token_masks)
    
                if self.prompt_length > 1:
                    updated_tgt2 = self.decoder2(tgt2, pos2, h_token, mask=token_masks)
                else:
                    updated_tgt2 = updated_tgt
                    
            intermediate.append({"h_token":h_token, "left_h_locator":updated_tgt, "right_h_locator":updated_tgt, "h_cls":updated_tgt2})

        output = []
        
        for h_dict in intermediate:
            h_token, left_h_locator, right_h_locator, h_cls = h_dict["h_token"], h_dict["left_h_locator"], h_dict["right_h_locator"], h_dict["h_cls"]
            p_left = self.left_boundary_classfier(h_token, left_h_locator, token_masks)
            p_right = self.right_boundary_classfier(h_token, right_h_locator, token_masks)

            #             # Apply Context-Aware Entity Enhancement(ASRN) here
            enhanced_entity_embeddings = self.context_aware_enhancer(h_token, h_token, token_masks)

            entity_logits = self.entity_classifier(h_cls)
            output.append({"p_left": p_left, "p_right": p_right, "entity_logits": entity_logits})

        return entity_logits, p_left, p_right, masked_seq_logits, output
    
    def _forward_train(self, *args, epoch=0, **kwargs):
        if not self.has_changed and epoch >= self.split_epoch and not self.freeze_transformer:
            logger.info(f"Now, update bert weights @ epoch = {self.split_epoch }")
            self.has_changed = True
            for name, param in self.named_parameters():
                param.requires_grad = True

        return self._common_forward(*args, **kwargs)

    def _forward_eval(self, *args, **kwargs):
        return self._common_forward(*args, **kwargs)

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)

class Bertpromptner(promptner):
    
    config_class = BertConfig
    base_model_prefix = "bert"
    # base_model_prefix = "model"
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, *args, **kwagrs):
        super().__init__("bert", *args, **kwagrs)

class Robertapromptner(promptner):

    config_class = RobertaConfig
    base_model_prefix = "roberta"
    # base_model_prefix = "model"
    
    def __init__(self, *args, **kwagrs):
        super().__init__("roberta", *args, **kwagrs)
    
class XLMRobertapromptner(promptner):

    config_class = XLMRobertaConfig
    base_model_prefix = "roberta"
    # base_model_prefix = "model"
    
    def __init__(self, *args, **kwagrs):
        super().__init__("roberta", *args, **kwagrs)

class Albertpromptner(promptner):

    config_class = XLMRobertaConfig
    base_model_prefix = "albert"
    # base_model_prefix = "model"
    
    def __init__(self, *args, **kwagrs):
        super().__init__("albert", *args, **kwagrs)


_MODELS = {
    'promptner': Bertpromptner,
    'roberta_promptner': Robertapromptner,
    'xlmroberta_promptner': XLMRobertapromptner,
    'albert_promptner': Albertpromptner
}

def get_model(name):
    return _MODELS[name]
