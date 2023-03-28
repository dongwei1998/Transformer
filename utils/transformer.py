# coding=utf-8
# =============================================
# @Time      : 2023-02-22 10:15
# @Author    : DongWei1998
# @FileName  : transformer.py
# @Software  : PyCharm
# =============================================
import os

import numpy as np
import tensorflow as tf
from utils import data_help, parameter


# 位置编码
def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


# 嵌入词表+位置编码
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model":self.d_model,
            "vocab_size":self.vocab_size,
        })
        return config

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x



# 基础的注意力
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


# 交叉的注意力
class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


# 全局的子注意力 self Attention
class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


# 因果注意力
class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


# 前馈网络
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        self.d_model = d_model
        self.dff = dff
        self.dropout_rate = dropout_rate
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model":self.d_model,
            "dff":self.dff,
            "dropout_rate":self.dropout_rate
        })
        return config

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


# 编码器层
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model":self.d_model,
            "num_heads":self.num_heads,
            "dff":self.dff,
            "dropout_rate":self.dropout_rate
        })
        return config

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


# 编码器
class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                 dff, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate

        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model":self.d_model,
            "num_layers":self.num_layers,
            "num_heads":self.num_heads,
            "dff":self.dff,
            "vocab_size":self.vocab_size,
            "dropout_rate":self.dropout_rate
        })
        return config

    def call(self, x,training):
        # print(" 编码器"+"输入:"+str(x.shape))
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x,training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


# 解码器层
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 d_model,
                 num_heads,
                 dff,
                 dropout_rate=0.1):
        super(DecoderLayer, self).__init__()



        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)



    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model":self.d_model,
            "num_heads":self.num_heads,
            "dff":self.dff,
            "dropout_rate":self.dropout_rate
        })
        return config


    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x


# 解码器
class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
                 dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                 d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model":self.d_model,
            "num_layers":self.num_layers,
            "num_heads":self.num_heads,
            "dff":self.dff,
            "vocab_size":self.vocab_size,
            "dropout_rate":self.dropout_rate,
            "last_attn_scores":self.last_attn_scores
        })
        return config

    def call(self, x, context,training):
        # `x` is token-IDs shape (batch, target_seq_len)
        #
        # print(" 解码器"+"输入:"+str(x.shape) +
        #       " 解码序列长度:"+str(len(context))+
        #       " 输入的上下文:"+str(context.shape) +
        #       " 上下文的长度:"+str(len(context[0])))
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)


        x = self.dropout(x,training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x


# Transformer模型
class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.dropout_rate = dropout_rate

        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)





    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model":self.d_model,
            "num_layers":self.num_layers,
            "num_heads":self.num_heads,
            "dff":self.dff,
            "input_vocab_size":self.input_vocab_size,
            "target_vocab_size":self.target_vocab_size,
            "dropout_rate":self.dropout_rate
        })
        return config


    def call(self, inputs,training=True):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context,x = inputs

        context = self.encoder(context,training)  # (batch_size, context_len, d_model)

        x = self.decoder(x, context,training)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        # Return the final output and the attention weights.
        return logits


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # 超参数
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    # 参数加载
    args = parameter.parser_opt()
    # 数据加载
    inp, targ = data_help.load_data()
    # 数据预处理 序列化工具加载
    if os.path.exists(args.input_vocab) and os.path.exists(args.target_vocab):
        input_text_processor = data_help.load_text_processor(args.input_vocab, args.max_seq_length)
        output_text_processor = data_help.load_text_processor(args.target_vocab, args.max_seq_length)
    else:
        input_text_processor = data_help.create_text_processor(args.input_vocab, inp, args.max_seq_length)
        output_text_processor = data_help.create_text_processor(args.target_vocab, targ, args.max_seq_length)

    # 原数据据打乱 批次化
    dataset = tf.data.Dataset.from_tensor_slices((inp, targ))
    dataset = dataset.batch(args.batch_size)
    # 获取词表大小
    args.input_vocab_size = input_text_processor.vocabulary_size()
    args.target_vocab_size = output_text_processor.vocabulary_size()

    # 模型构建
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=args.input_vocab_size,
        target_vocab_size=args.target_vocab_size,
        dropout_rate=dropout_rate)


    # 构建预训练的模型
    for batch, (example_input_batch, example_target_batch) in enumerate(dataset):
        # 数据转换id
        input_tokens = input_text_processor(example_input_batch)
        target_tokens = output_text_processor(example_target_batch)

        # 训练模型
        output = transformer((input_tokens, target_tokens))

        # 模型可视化
        transformer.summary()
        print("输入: %s\n" % input_text_processor(example_input_batch))
        print("输出: %s\n" % output.shape)

        break
