import argparse

from args import train_argparser, eval_argparser
from config_reader import process_configs
from promptner import input_reader
from promptner.promptner_trainer import PromptNERTrainer
import warnings

import logging
# 配置日志基本设置
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


def __train(run_args):
    logger.info("开始训练过程")
    trainer = PromptNERTrainer(run_args)
    trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)
    logger.info("训练过程结束")


def _train():
    logger.info("初始化训练参数解析器")
    arg_parser = train_argparser()
    logger.info("开始处理训练配置")
    process_configs(target=__train, arg_parser=arg_parser)
    logger.info("训练配置处理完成")


def __eval(run_args):
    logger.info("开始评估过程")
    trainer = PromptNERTrainer(run_args)
    trainer.eval(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                 input_reader_cls=input_reader.JsonInputReader)
    logger.info("评估过程结束")


def _eval():
    logger.info("初始化评估参数解析器")
    arg_parser = eval_argparser()
    logger.info("开始处理评估配置")
    process_configs(target=__eval, arg_parser=arg_parser)
    logger.info("评估配置处理完成")


if __name__ == '__main__':
    logger.info("程序启动")
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")
    args, _ = arg_parser.parse_known_args()

    if args.mode == 'train':
        logger.info("进入训练模式")
        _train()
    elif args.mode == 'eval':
        logger.info("进入评估模式")
        _eval()
    else:
        logger.error("无效的模式参数: %s", args.mode)
        raise Exception("Mode not in ['train', 'eval'], e.g. 'python promptner.py train ...'")
