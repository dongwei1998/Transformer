# coding=utf-8
# =============================================
# @Time      : 2023-04-01 14:53
# @Author    : DongWei1998
# @FileName  : bloak_test.py
# @Software  : PyCharm
# =============================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 指定第一块GPU可用

import tensorflow as tf
print(tf.data.AUTOTUNE)


# 推理测试
reloaded = tf.saved_model.load('./model_2')
text = 'este é o primeiro livro que eu fiz.'
result = reloaded(tf.constant(text))
print(result)