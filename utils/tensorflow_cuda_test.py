# coding=utf-8
# =============================================
# @Time      : 2023-02-21 11:07
# @Author    : DongWei1998
# @FileName  : tensorflow_cuda_test.py
# @Software  : PyCharm
# =============================================y


import tensorflow as tf
tf.config.list_physical_devices('GPU')


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)


tf.test.is_built_with_cuda()

