import tensorflow as tf
import scipy.misc
import numpy as np


def print_network():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)


def denorm(image):
    image = (image + 1) / 2
    return np.clip(image, 0, 1)


def normalize(x):
    return x / 127.5 - 1
