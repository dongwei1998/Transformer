# coding=utf-8
# =============================================
# @Time      : 2023-02-15 16:13
# @Author    : DongWei1998
# @FileName  : train.py
# @Software  : PyCharm
# =============================================
import os

import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import tensorflow as tf
from utils import parameter,data_help,gpu_git
from matplotlib import pyplot as plt
import time
from utils.transformer import Transformer
from utils.token_tool import Tokenizers,standardize




# 定义目标函数
def loss_fun(y_ture, y_pred,loss_object):
    mask = tf.math.logical_not(tf.math.equal(y_ture, 0))  # 为0掩码标1
    loss_ = loss_object(y_ture, y_pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)




# 优化器
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# 动态学习率测试
def CustomSchedule_test(args):
    temp_learing_rate = CustomSchedule(args.embedding_size)
    plt.plot(temp_learing_rate(tf.range(40000, dtype=tf.float32)))
    plt.xlabel('train step')
    plt.ylabel('learing rate')
    plt.show()


def print_translation(sentence, tokens, ground_truth):
  print(f'{"Input:":15s}: {sentence}')
  print(f'{"Prediction":15s}: {tokens}')
  print(f'{"Ground truth":15s}: {ground_truth}')


class ExportTranslator(tf.Module):
  def __init__(self, translator):
    self.translator = translator

  @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  def __call__(self, sentence):
    (result,
     tokens,
     attention_weights) = self.translator(sentence, max_length=args.max_seq_length)

    return result




def train_model(args):
    # 数据加载
    inp, targ,targ_label = data_help.load_data()

    # 数据预处理 序列化工具加载
    if os.path.exists(args.input_vocab) and os.path.exists(args.target_vocab):
        token_tool_a = Tokenizers(input_vocab=args.input_vocab,max_length=args.max_seq_length,standardize=standardize)
        token_tool_b = Tokenizers(input_vocab=args.target_vocab, max_length=args.max_seq_length, standardize=standardize)
    else:
        token_tool_a = Tokenizers(input_vocab=args.input_vocab,data=inp,max_length=args.max_seq_length, standardize=standardize)
        token_tool_b = Tokenizers(input_vocab=args.target_vocab, data=targ, max_length=args.max_seq_length,standardize=standardize)

    # 获取词表大小
    args.input_vocab_size = token_tool_a.tokenizers.vocabulary_size()
    args.target_vocab_size = token_tool_b.tokenizers.vocabulary_size()

    inp_ids = token_tool_a.text_to_ids(inp)
    targ_ids = token_tool_b.text_to_ids(targ)
    targ_label_ids = token_tool_b.text_to_ids(targ_label)


    dataset = tf.data.Dataset.from_tensor_slices((inp_ids, targ_ids,targ_label_ids))
    dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(args.batch_size)


    # 数据可视化
    for batch, (inp_ids, targ_ids, targ_label_ids) in enumerate(dataset):
        print("原始读取的 文本 数据...")
        print(inp[:args.batch_size])
        print(targ[:args.batch_size])
        print(targ_label[:args.batch_size])
        print("数据预处理后的 idx 数据...")
        print(inp_ids)
        print(targ_ids)
        print(targ_label_ids)
        print("数据预处理后的 文本 数据...")
        print(token_tool_a.ids_to_text(inp_ids))
        print(token_tool_b.ids_to_text(targ_ids))
        print(token_tool_b.ids_to_text(targ_label_ids))
        break

    # 模型构建
    transformer = Transformer(
        num_layers=args.num_layers,
        d_model=args.embedding_size,
        num_heads=args.num_heads,
        dff=args.feed_input_size,
        input_vocab_size=args.input_vocab_size,
        target_vocab_size=args.target_vocab_size,
        dropout_rate=args.dropout_rate)

    # 模型优化器
    learing_rate = CustomSchedule(args.embedding_size)
    optimizer = tf.keras.optimizers.Adam(learing_rate, beta_1=0.9,beta_2=0.98, epsilon=1e-9)

    # 定义目标函数
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


    # 模型保存器
    if os.path.exists(args.checkpoint_path):
        print(f'load model {args.checkpoint_path}')
        tf.keras.models.load_model(args.checkpoint_path, custom_objects={'transformer': transformer})




    step = 0
    step_list = []
    loss_list = []
    for epoch in range(args.num_epochs):
        start_s = time.time()
        # 重置记录项
        train_loss.reset_states()
        train_accuracy.reset_states()
        for batch, (inp_ids, targ_ids,targ_label_ids) in enumerate(dataset):
            with tf.GradientTape() as tape:
                logits = transformer((inp_ids, targ_ids))
                loss = loss_fun(targ_label_ids, logits, loss_object)
            # 求梯度
            gradients = tape.gradient(loss, transformer.trainable_variables)
            # 反向传播
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
            # 记录loss和准确率
            train_loss(loss)
            train_accuracy(targ_label_ids, logits)
            if step % args.step_env_model == 0:
                loss = train_loss.result()
                print('epoch {}, batch {}, loss:{:.4f}, acc:{:.4f}'.format(
                    epoch + 1, batch, loss, train_accuracy.result()
                ))
                step_list.append(step)
                loss_list.append(loss)

            # custom_objects = {'CustomLayer': CustomLayer,
            #                   'custom_activation': custom_activation}
            # with keras.utils.custom_object_scope(custom_objects):
            #     new_model = keras.Model.from_config(config)


            if (step + 1) % args.ckpt_model_num == 0:
                tf.keras.models.save_model(transformer, args.checkpoint_path)
                print('epoch {}, save model at {}'.format(
                    epoch + 1, args.checkpoint_path
                ))
            step+=1
            print(step)

        print('epoch {}, loss:{:.4f}, acc:{:.4f}'.format(
            epoch + 1, train_loss.result(), train_accuracy.result()
        ))
        print('time in 1 epoch:{} secs\n'.format(time.time() - start_s))



    # 打印loss
    plt.plot(step_list, loss_list)
    plt.xlabel('train step')
    plt.ylabel('loss')
    plt.savefig(f'./train_loss.jpg')

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 指定第一块GPU可用
    # 参数加载
    args = parameter.parser_opt()
    # gpu选择策略
    # gpu_git.check_gpus(mode=1, logger=args.logger)
    # 训练
    train_model(args)




