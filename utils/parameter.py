# coding=utf-8
# =============================================
# @Time      : 2023-02-17 16:15
# @Author    : DongWei1998
# @FileName  : parameter.py
# @Software  : PyCharm
# =============================================
import os
from easydict import EasyDict
from dotenv import load_dotenv,find_dotenv
import logging.config

# 创建路径
def check_directory(path, create=True):
    flag = os.path.exists(path)
    if not flag:
        if create:
            os.makedirs(path)
            flag = True
    return flag


def parser_opt():
    load_dotenv(find_dotenv())  # 将.env文件中的变量加载到环境变量中
    args = EasyDict()
    logging.config.fileConfig(os.environ.get("logging_ini"))
    args.logger = logging.getLogger('model_log')
    args.checkpoint_path = os.environ.get('checkpoint_path')
    args.logging_ini = os.environ.get('logging_ini')
    args.train_data_file = os.environ.get('train_data_file')
    args.test_data_file = os.environ.get('test_data_file')
    args.val_data_file = os.environ.get('val_data_file')
    args.input_vocab = os.environ.get('input_vocab')
    args.target_vocab = os.environ.get('target_vocab')
    args.batch_size = int(os.environ.get('batch_size'))
    args.num_epochs = int(os.environ.get('num_epochs'))
    args.num_layers = int(os.environ.get('num_layers'))
    args.num_heads = int(os.environ.get('num_heads'))
    args.input_vocab_size = int(os.environ.get('input_vocab_size'))
    args.target_vocab_size = int(os.environ.get('target_vocab_size'))
    args.embedding_size = int(os.environ.get('embedding_size'))
    args.dropout_rate = float(os.environ.get('dropout_rate'))
    args.feed_input_size = int(os.environ.get('feed_input_size'))
    args.max_seq_length = int(os.environ.get('max_seq_length'))
    args.ckpt_model_num = int(os.environ.get('ckpt_model_num'))
    args.step_env_model = int(os.environ.get('step_env_model'))
    args.model_ckpt_name = os.environ.get('model_ckpt_name')


    return args