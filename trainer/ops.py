import tensorflow as tf
from tensorflow import layers


def deconv2d(x, filters, kernels=(5, 5), strides=(2, 2), padding='same', training=False):
    x = layers.conv2d_transpose(x, filters, kernels, strides, padding,
                                kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02))
    # x = layers.batch_normalization(x, momentum=0.9, training=training)
    x = tf.nn.leaky_relu(x)
    return x


def conv2d(x, filter, kernels=(5, 5), strides=(2, 2), padding='same', training=False):
    x = layers.conv2d(x, filter, kernels, strides, padding,
                      kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02))
    # x = layers.batch_normalization(x, momentum=0.9, training=training)
    x = tf.nn.leaky_relu(x)
    return x
