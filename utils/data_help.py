# coding=utf-8
# =============================================
# @Time      : 2022-09-22 9:42
# @Author    : DongWei1998
# @FileName  : data_help.py
# @Software  : PyCharm
# =============================================
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf
import os




# 数据加载
def load_data():
    # 葡萄牙语->英语
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']
    inp = []
    targ = []
    for pt, en in train_examples:
        inp.append(pt.numpy())
        targ.append(en.numpy())
    return inp, targ



# 编码规范化 数据预处理   西班牙语 英文
def tf_lower_and_split_punct(text):
    '''
    ¿Todavía está en casa?
    [START] ¿ todavia esta en casa ? [END]
    '''
    # # 分离重音字符
    # text = tf_text.normalize_utf8(text, 'NFKD')
    # text = tf.strings.lower(text)
    # 保留空格 a-z 标点符号.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # 标点符号周围添加空格.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # 分割添加开始结束标志.
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text

# 序列化工具
def text_processor(max_seq_length,vocabulary=None):
    if vocabulary == None:
        return tf.keras.layers.experimental.preprocessing.TextVectorization(
            standardize=tf_lower_and_split_punct, # 数据预处理函数，根据业务场景修改
            output_sequence_length=max_seq_length
        )
    return tf.keras.layers.experimental.preprocessing.TextVectorization(
            standardize=tf_lower_and_split_punct,
            vocabulary=vocabulary,
            output_sequence_length=max_seq_length
    )


def load_text_processor(vocab_file_path,max_seq_length):
    with open(vocab_file_path, 'r', encoding='utf-8') as ir:
        vocabulary = ir.read().split('\n')
    vocab_text_processor = text_processor(max_seq_length, vocabulary=vocabulary)
    return vocab_text_processor


def create_text_processor(vocab_file_path,vocab_list,max_seq_length):
    vocab_text_processor = text_processor(max_seq_length)
    vocab_text_processor.adapt(vocab_list)
    vocabulary = vocab_text_processor.get_vocabulary()
    with open(vocab_file_path, 'w', encoding='utf-8') as iw:
        iw.write('\n'.join(vocabulary))
    return vocab_text_processor

def preprocess(input_text, target_text):

    # Convert the text to token IDs
    input_tokens = input_text_processor(input_text)
    target_tokens = output_text_processor(target_text)

    # Convert IDs to masks.
    input_mask = input_tokens != 0
    target_mask = target_tokens != 0

    return input_tokens, input_mask, target_tokens, target_mask


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
    inp, targ = load_data()

    input_vocab = '../config/input_vocab.txt'
    output_vocab = '../config/output_vocab.txt'
    max_seq_length = 250
    batch_size = 4

    # 数据预处理 序列化工具加载
    if os.path.exists(input_vocab) and os.path.exists(output_vocab):
        input_text_processor = load_text_processor(input_vocab,max_seq_length)
        output_text_processor = load_text_processor(output_vocab,max_seq_length)
    else:
        input_text_processor = create_text_processor(input_vocab,inp,max_seq_length)
        output_text_processor = create_text_processor(output_vocab,targ,max_seq_length)

    # 原数据据打乱 批次化
    dataset = tf.data.Dataset.from_tensor_slices((inp, targ))
    dataset = dataset.batch(batch_size)

    for example_input_batch, example_target_batch in dataset:
        # 数据预处理 序列化工具
        (input_tokens, input_mask, target_tokens, target_mask) = preprocess(example_input_batch, example_target_batch)
        # 获取序列长度
        max_target_length = tf.shape(target_tokens)[1]
        print(input_tokens, input_mask, target_tokens, target_mask)
        break



