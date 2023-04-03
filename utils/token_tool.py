# coding=utf-8
# =============================================
# @Time      : 2023-03-23 16:37
# @Author    : DongWei1998
# @FileName  : token_tool.py
# @Software  : PyCharm
# =============================================
import os

import numpy as np
import tensorflow as tf

class Tokenizers(tf.keras.layers.TextVectorization):
    def __init__(self, input_vocab=None,data=None,standardize=None,max_length=None, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        if data is not None:
            self.tokenizers = self.text_processor(standardize=standardize)
            with open(input_vocab, 'w', encoding='utf-8') as w:
                data = data+ ["[UNK] [START] [END]"]
                self.tokenizers.adapt(data)
                vocab_list = self.tokenizers.get_vocabulary()
                w.write('\n'.join(vocab_list))
        else:
            # 有词表
            with open(input_vocab, 'r', encoding='utf-8') as r:
                vocab_list = [word.replace('\n', '') for word in r.readlines()]
                self.tokenizers = self.text_processor(vocabulary=vocab_list, standardize=standardize)


        self.text_to_ids_dict = {word:idx for idx,word in enumerate(vocab_list)}
        self.ids_to_text_dict = {v: k for k, v in self.text_to_ids_dict.items()}

    def get_config(self):
        config = super().get_config()
        config.update({
            "tokenizers":self.tokenizers,
            "max_length": self.max_length,
            "text_to_ids_dict":self.text_to_ids_dict,
            "ids_to_text_dict":self.ids_to_text_dict,
        })
        return config

    # 构建序列化工具
    def text_processor(self,vocabulary=None,standardize=None):
        if vocabulary == None:
            return tf.keras.layers.TextVectorization(
                standardize=standardize,  # 数据预处理函数，根据业务场景修改
                output_sequence_length=self.max_length
            )
        return tf.keras.layers.TextVectorization(
            standardize=standardize,
            vocabulary=vocabulary,
            output_sequence_length=self.max_length
        )




    def text_to_ids(self,sens):
        # _ids_list = []
        # for sen in sens:
        #     _ids = []
        #     for i,word in enumerate(sen.split(' ')):
        #         if word in self.text_to_ids_dict.keys():
        #             _ids.append(str(self.text_to_ids_dict[word]))
        #         else:
        #             _ids.append(str(self.text_to_ids_dict['[UNK]']))
        #     _ids_list.append(" ".join(_ids))
        # return _ids_list

        return self.tokenizers(sens)


    def ids_to_text(self,ids_s):
        return ids_s
        # _word_list = []
        # for ids in ids_s:
        #     _word = []
        #     # 去除填充的0
        #     for _id in ids.numpy():
        #         if _id in self.ids_to_text_dict.keys():
        #             if _id != 0:
        #                 _word.append(self.ids_to_text_dict[_id])
        #         else:
        #             _word.append('[UNK]')
        #     _word_list.append(' '.join(_word))
        #
        # return _word_list




def standardize(text):
    # 编码规范化 数据预处理   西班牙语 英文
    '''
    ¿Todavía está en casa?
    [START] ¿ todavia esta en casa ? [END]
    '''
    # 去除 无用的标点符号
    # text.replace('``', '').replace("''", '')
    # # 分离重音字符
    # text = tf_text.normalize_utf8(text, 'NFKD')
    # text = tf.strings.lower(text)
    # # 保留空格 a-z 标点符号.
    # text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # # 标点符号周围添加空格.
    # text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # 分割添加开始结束标志.
    # text = tf.strings.strip(text)
    # text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text
