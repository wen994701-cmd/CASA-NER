# CASA-NER

<p float="left">
  <img src="./assets/comparison.png" width="28.7%" />
  <img src="./assets/overview.png" width="69%" />
</p>

## 设置

本项目基于 Python 3.8 环境运行，所需依赖见 requirements.txt 文件。

### 数据

论文实验涉及的数据集包括：

ACE2004

CoNLL2003

Genia

BC5CDR

### 训练与评估说明

模型的训练与评估统一通过 train_eval.py 进行控制。

训练与测试过程由配置文件进行参数管理

各数据集实验设置对应不同 .conf 文件

所有关键超参数均在配置文件中进行统一定义

