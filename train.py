# coding=utf-8
# =============================================
# @Time      : 2023-02-15 16:13
# @Author    : DongWei1998
# @FileName  : train.py
# @Software  : PyCharm
# =============================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import tensorflow as tf
from utils import parameter,data_help,gpu_git
from matplotlib import pyplot as plt
import time
from utils.transformer import Transformer




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




def train_step(inputs, targets,transformer_model,loss_object,optimizer,train_loss,train_accuracy):

    tar_inp = targets[:,:-1]    # Drop the [END] tokens
    tar_real = targets[:,1:]    # Drop the [START] tokens

    with tf.GradientTape() as tape:
        logits = transformer_model((inputs,tar_inp))
        loss = loss_fun(tar_real, logits,loss_object)
    # 求梯度
    gradients = tape.gradient(loss, transformer_model.trainable_variables)
    # 反向传播
    optimizer.apply_gradients(zip(gradients, transformer_model.trainable_variables))

    # 记录loss和准确率
    train_loss(loss)
    train_accuracy(tar_real, logits)


def train_model(args):
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
        num_layers=args.num_layers,
        d_model=args.embedding_size,
        num_heads=args.num_heads,
        dff=args.feed_input_size,
        input_vocab_size=args.input_vocab_size,
        target_vocab_size=args.target_vocab_size,
        dropout_rate=args.dropout_rate)


    # 模型优化器
    learing_rate = CustomSchedule(args.embedding_size)
    optimizer = tf.keras.optimizers.Adam(learing_rate, beta_1=0.9,
                                         beta_2=0.98, epsilon=1e-9)

    # 定义目标函数
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction='none')


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # 模型保存器
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    # ckpt管理器
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.checkpoint_path, max_to_keep=3)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('last checkpoit restore')
    step = 0
    step_list = []
    loss_list = []
    # 模型训练
    for epoch in range(args.num_epochs):
        start = time.time()
        # 重置记录项
        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch, (example_input_batch, example_target_batch) in enumerate(dataset):
            # Convert the text to token IDs
            input_tokens = input_text_processor(example_input_batch)
            target_tokens = output_text_processor(example_target_batch)
            # example_input_batch 葡萄牙语， example_target_batch
            train_step(
                inputs = input_tokens,
                targets = target_tokens,
                transformer_model = transformer,
                loss_object = loss_object,
                optimizer = optimizer,
                train_loss = train_loss,
                train_accuracy = train_accuracy)
            if batch % 500 == 0:
                loss = train_loss.result()
                print('epoch {}, batch {}, loss:{:.4f}, acc:{:.4f}'.format(
                    epoch + 1, batch, loss, train_accuracy.result()
                ))
                step_list.append(step)
                loss_list.append(loss)
            step += 1
        if (epoch + 1) % 2 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('epoch {}, save model at {}'.format(
                epoch + 1, ckpt_save_path
            ))

        print('epoch {}, loss:{:.4f}, acc:{:.4f}'.format(
            epoch + 1, train_loss.result(), train_accuracy.result()
        ))

        print('time in 1 epoch:{} secs\n'.format(time.time() - start))
    # 打印loss
    plt.plot(step_list, loss_list)
    plt.xlabel('train step')
    plt.ylabel('loss')
    plt.savefig(f'./train_loss.jpg')

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 指定第一块GPU可用
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu80%的显存
    config.gpu_options.allow_growth = True  # 程序按需申请内存
    sess = tf.compat.v1.Session(config=config)
    # 参数加载
    args = parameter.parser_opt()
    # gpu选择策略
    # gpu_git.check_gpus(mode=0, logger=args.logger)
    # 训练
    train_model(args)

    # 测试动态学习率
    # CustomSchedule_test(args)

