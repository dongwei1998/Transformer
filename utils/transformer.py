# coding=utf-8
# =============================================
# @Time      : 2023-02-15 16:15
# @Author    : DongWei1998
# @FileName  : transformer.py
# @Software  : PyCharm
# =============================================
import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 只显示 warning 和 Error 等级的信息，不显示具体的信息
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # 在 GPU0上运行此代码，显存被挤出


import tensorflow as tf
from tensorflow import keras




class Encoder(keras.layers.Layer):
    def __init__(self, num_layers, embedding_size, num_heads, feed_input_size,input_vocab_size, max_seq_length, dropout_rate):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        # 嵌入词表
        self.embedding = keras.layers.Embedding(input_vocab_size, embedding_size)
        # 位置编码
        self.pos_embedding = positional_encoding(max_seq_length, embedding_size)
        # 编码器
        self.encode_layer = [TransformerEncoder(num_heads, feed_input_size, embedding_size, dropout_rate) for _ in range(num_layers)]
        #  Dropout层
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training, mark):

        # 获取序列长度
        seq_len = inputs.shape[1]
        # embedding 嵌入词表
        word_emb = self.embedding(inputs)
        word_emb *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32))
        emb = word_emb + self.pos_embedding[:,:seq_len,:]
        x = self.dropout(emb, training=training)

        for i in range(self.num_layers):
            x = self.encode_layer[i](x, training, mark)
        return x

# 归一化
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        self.eps = epsilon
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=tf.ones_initializer(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=tf.zeros_initializer(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape



# TransformerEncoder
class TransformerEncoder(keras.layers.Layer):
    def __init__(self, num_heads, feed_input_size, embedding_size, dropout_rate):
        super(TransformerEncoder, self).__init__()

        self.mha = MultiHeadAttention(embedding_size, num_heads)
        self.ffn = point_wise_feed_forward_network(embedding_size, feed_input_size)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)


        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        # 多头注意力网络
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        # 残差+归一化
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        # 前向网络
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        # 残差+归一化
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2



class Decoder(keras.layers.Layer):
    def __init__(self, num_layers, embedding_size, num_heads, feed_input_size, target_vocab_size, max_seq_length, drop_rate=0.1):
        super(Decoder, self).__init__()

        self.embedding_size = embedding_size
        self.num_layers = num_layers

        self.embedding = keras.layers.Embedding(target_vocab_size, self.embedding_size)
        self.pos_embedding = positional_encoding(max_seq_length, self.embedding_size)

        self.decoder_layers= [TransformerDecoder(self.embedding_size, num_heads, feed_input_size, drop_rate) for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(drop_rate)

    def call(self, inputs, encoder_out,training,look_ahead_mark, padding_mark):

        seq_len = tf.shape(inputs)[1]
        attention_weights = {}
        h = self.embedding(inputs)
        h *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32))
        h += self.pos_embedding[:,:seq_len,:]
        h = self.dropout(h, training=training)
        # 叠加解码层
        for i in range(self.num_layers):
            h, att_w1, att_w2 = self.decoder_layers[i](h, encoder_out,training, look_ahead_mark,padding_mark)
            attention_weights['decoder_layer{}_att_w1'.format(i+1)] = att_w1
            attention_weights['decoder_layer{}_att_w2'.format(i+1)] = att_w2
        return h, attention_weights

# TransformerDecoder
class TransformerDecoder(keras.layers.Layer):
    def __init__(self, embedding_size, num_heads, feed_input_size, dropout_rate):
        super(TransformerDecoder, self).__init__()
        self.mha1 = MultiHeadAttention(embedding_size, num_heads)
        self.mha2 = MultiHeadAttention(embedding_size, num_heads)
        self.ffn = point_wise_feed_forward_network(embedding_size, feed_input_size)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        self.dropout3 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, encode_out, training, look_ahead_mask, padding_mask):
        # masked muti-head attention
        att1, att_weight1 = self.mha1(inputs, inputs, inputs, look_ahead_mask)
        att1 = self.dropout1(att1, training=training)
        out1 = self.layernorm1(inputs + att1)
        # muti-head attention
        att2, att_weight2 = self.mha2(encode_out, encode_out, inputs, padding_mask)
        att2 = self.dropout2(att2, training=training)
        out2 = self.layernorm2(out1 + att2)

        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        out3 = self.layernorm3(out2 + ffn_out)

        return out3, att_weight1, att_weight2


# 点积计算注意力
def scaled_dot_product_attention(q, k, v, mask):
    # query key 相乘获取匹配关系
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # 使用dk进行缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 掩码
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 通过softmax获取attention权重
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # attention 乘上value
    output = tf.matmul(attention_weights, v)  # （.., seq_len_v, depth）

    return output, attention_weights


def create_look_ahead_mark(size):
    # 1 - 对角线和取下三角的全部对角线（-1->全部）
    # 这样就可以构造出每个时刻未预测token的掩码
    mark = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mark  # (seq_len, seq_len)

# 掩码（mask）
def create_padding_mark(seq):
    # 获取为0的padding项
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # 扩充维度以便用于attention矩阵
    return seq[:, np.newaxis, np.newaxis, :]

# 位置编码
def get_angles(pos, i, embedding_size):
    # 这里的i等价与上面公式中的2i和2i+1
    angle_rates = 1 / np.power(10000, (2*(i // 2))/ np.float32(embedding_size))
    return pos * angle_rates
def positional_encoding(max_seq_length, embedding_size):
    angle_rads = get_angles(np.arange(max_seq_length)[:, np.newaxis],
                           np.arange(embedding_size)[np.newaxis,:],
                           embedding_size)
    # 第2i项使用sin
    sines = np.sin(angle_rads[:, 0::2])
    # 第2i+1项使用cos
    cones = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cones], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

# 前馈网络
def point_wise_feed_forward_network(embedding_size, feed_input_size):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(feed_input_size, activation='relu'),
        tf.keras.layers.Dense(embedding_size)
    ])

# 构造mutil head attention层
class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, embedding_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_size = embedding_size

        # embedding_size 必须可以正确分为各个头
        assert embedding_size % num_heads == 0
        # 分头后的维度
        self.depth = embedding_size // num_heads

        self.wq = tf.keras.layers.Dense(embedding_size)
        self.wk = tf.keras.layers.Dense(embedding_size)
        self.wv = tf.keras.layers.Dense(embedding_size)

        self.dense = tf.keras.layers.Dense(embedding_size)

    def split_heads(self, x, batch_size):
        # 分头, 将头个数的维度 放到 seq_len 前面
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # 分头前的前向网络，获取q、k、v语义
        q = self.wq(q)  # (batch_size, seq_len, embedding_size)
        k = self.wk(k)
        v = self.wv(v)

        # 分头
        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        # 通过缩放点积注意力层
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # 把多头维度后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3]) # (batch_size, seq_len_v, num_heads, depth)
        # 合并多头
        concat_attention = tf.reshape(scaled_attention,(batch_size, -1, self.embedding_size))
        # 全连接重塑
        output = self.dense(concat_attention)
        return output, attention_weights


class Transformer(keras.Model):
    def __init__(self, num_layers, num_heads, feed_input_size, max_seq_length,input_vocab_size,target_vocab_size,embedding_size, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, embedding_size, num_heads, feed_input_size,input_vocab_size, max_seq_length, dropout_rate)
        self.decoder = Decoder(num_layers, embedding_size, num_heads, feed_input_size,target_vocab_size, max_seq_length, dropout_rate)
        self.dropout_layer = keras.layers.Dropout(dropout_rate)


        self.final_layer = keras.layers.Dense(target_vocab_size)



    def call(self, inputs, targets, training, encode_padding_mask, look_ahead_mask, decode_padding_mask):
        # 编码器
        encode_out = self.encoder(inputs, training, encode_padding_mask) # (batch_size, inp_seq_len, d_model)
        # 解码器
        decode_out, attention_weights = self.decoder(targets, encode_out, training, look_ahead_mask,decode_padding_mask)
        # 全连接 不做激活 输出的是logits
        final_output = self.final_layer(decode_out)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output, attention_weights





if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

    transformer = Transformer(
        num_layers=6,
        num_heads=8,
        input_vocab_size=9500,
        target_vocab_size=8000,
        embedding_size=256,
        dropout_rate=0.1,
        feed_input_size=1024,
        max_seq_length=312
    )


    # 1. 定义输入变量
    temp_input = tf.random.uniform((64, 25))
    temp_target = tf.random.uniform((64, 6))


    final_output, _ = transformer(
        inputs=temp_input,
        targets=temp_target,
        training=False,
        encode_padding_mask=None,
        look_ahead_mask=None,
        decode_padding_mask=None
    )
    print(final_output.shape)

    transformer.summary()











