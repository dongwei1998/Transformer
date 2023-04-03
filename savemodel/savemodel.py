# coding=utf-8
# =============================================
# @Time      : 2023-03-30 14:01
# @Author    : DongWei1998
# @FileName  : savemodel.py
# @Software  : PyCharm
# =============================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from tensorflow import keras
import tensorflow as tf
import numpy as np

class CustomLayer(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(CustomLayer, self).get_config()
        config.update({"units": self.units})
        return config


def custom_activation(x):
    return tf.nn.tanh(x) ** 2


# Make a model with the CustomLayer and custom_activation
inputs = keras.Input((32,))
x = CustomLayer(32)(inputs)
outputs = keras.layers.Activation(custom_activation)(x)
model = keras.Model(inputs, outputs)
print(model)
# Retrieve the config
config = model.get_config()
serialized_layer = keras.layers.serialize(model)
print(serialized_layer)
# At loading time, register the custom objects with a `custom_object_scope`:
custom_objects = {"CustomLayer": CustomLayer, "custom_activation": custom_activation}
with keras.utils.custom_object_scope(custom_objects):
    new_model = keras.Model.from_config(config)
    print(new_model)