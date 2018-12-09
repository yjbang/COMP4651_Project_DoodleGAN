from base import BaseModel
from trainer.ops import deconv2d, conv2d
from trainer.input_fn import input_fn
from utils import print_network

import tensorflow as tf

tfgan = tf.contrib.gan
from tensorflow import layers


class GAN(BaseModel):
    def __init__(self, config, inputs):
        """DoodleGAN
        Args:
            config: (parameters) model hyperparameters
            is_training: (bool) whether we are training or not
            inputs: (dict) contains the inputs of the graph (features, labels...)
                    this can be 'tf.placeholder' or outputs of 'tf.data'
        """
        super(GAN, self).__init__(config)  # configs/baseline.json
        self.real_image, self.real_labels = inputs

        self.g_dim = config.g_dim
        self.g_layers = config.g_layers
        self.d_dim = config.d_dim
        self.d_layers = config.d_layers
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.g_learning_rate = config.g_learning_rate
        self.d_learning_rate = config.d_learning_rate
        self.image_size = config.image_size
        self.build_model()
        self.init_saver()

    def build_model(self):
        noise = tf.random_normal([self.config.batch_size, 100])

        wrong_image = tf.random.shuffle(self.real_image)
        # fake_labels = tf.random.uniform([self.config.batch_size], 0, self.config.num_classes, dtype=tf.int32)
        # fake_labels = tf.one_hot(fake_labels, self.config.num_classes)

        # output of D for real image
        real_logit = self.discriminator(self.real_image, self.real_labels, training=True)

        # output of D for wrong caption/image
        wrong_logit = self.discriminator(wrong_image, self.real_labels, training=True, reuse=True)

        # output of D for fake image
        self.fake_image = self.generator(noise, self.real_labels, training=True)
        fake_logit = self.discriminator(self.fake_image, self.real_labels, training=True, reuse=True)

        # ops for sampling an image
        sample_noise = tf.random_normal([3, 100], 0, 0.3)  # use smaller deviation noise for higher-quality image
        sample_label = tf.one_hot([0, 1, 2], self.config.num_classes)
        self.sample_image = self.generator(sample_noise, sample_label, training=False, reuse=True)

        """(Reed, 2016 "Generative Adversarial Text to Image Synthesis" GAN-CLS Loss)"""
        # get loss for discriminator
        self.d_loss1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit, labels=tf.ones_like(real_logit)),
            name="d_real_loss")
        self.d_loss2 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=wrong_logit, labels=tf.zeros_like(wrong_logit)),
            name="d_wrong_loss")
        self.d_loss3 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.zeros_like(fake_logit)),
            name="d_fake_loss")
        self.d_loss = self.d_loss1 + (self.d_loss2 + self.d_loss3) * 0.5

        # get loss for generator
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.ones_like(fake_logit)),
            name="g_fake_loss")

        # Optimizers
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        self.d_optim = tf.train.AdamOptimizer(self.config.d_learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(
            self.d_loss, var_list=d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.config.g_learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(
            self.g_loss, var_list=g_vars)

    def generator(self, noise, label, training, reuse=False, ):
        with tf.variable_scope('generator', reuse=reuse):
            net = tf.concat([label, noise], axis=-1)  # (batch_size, 445)
            net = layers.dense(net, self.g_dim * 3 * 3 * 4,  # (batch_size, 32768)
                               kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02))
            net = tf.nn.leaky_relu(net)
            net = tf.reshape(net, [-1, 3, 3, self.g_dim * 4])  # Reshape (3 x 3 x 512)
            net = deconv2d(net, self.g_dim * 2, strides=[1, 1], padding='valid', training=training)  # (7 x 7 x 256)
            net = deconv2d(net, self.g_dim, training=training)  # (14 x 14 x 128)
            net = layers.conv2d_transpose(net, 1, kernel_size=(5, 5), strides=(2, 2),
                                          activation=tf.nn.tanh, padding='same',
                                          kernel_initializer=tf.truncated_normal_initializer(
                                              mean=0, stddev=0.02))  # (28 x 28 x 1)
            return net  # (28, 28, 1)

    def discriminator(self, image, conditioning, training, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            net = conv2d(image, self.d_dim, training=training)  # (14 x 14 x 64)
            net = conv2d(net, self.d_dim * 2, training=training)  # (7 x 7 x 128)
            net = conv2d(net, self.d_dim * 4, training=training)  # (4 x 4 x 128)

            conditioning = tf.reshape(conditioning, [-1, 1, 1, conditioning.shape[1]])
            conditioning = tf.tile(conditioning, [1, net.shape[1], net.shape[2], 1])
            net = tf.concat([conditioning, net], axis=-1)  # (4 x 4 x 601)

            net = conv2d(net, self.d_dim * 4, kernels=1, strides=1)  # (4 x 4 x 256)
            net = layers.conv2d(net, 1, kernel_size=4, strides=1)  # (1 x 1 x 1)
            net = tf.nn.sigmoid(net)
            net = tf.squeeze(net)
            return net

    def init_saver(self):
        with tf.name_scope("saver"):
            self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)


def get_input_fn(filename):
    return lambda: input_fn(filename)


def build_estimator(model_dir):
    return tfgan.estimator.GANEstimator(
        model_dir,
        generator_fn=GAN.generator,
        discriminator_fn=GAN.discriminator,
        generator_loss_fn=tfgan.losses.modified_generator_loss,
        discriminator_loss_fn=tfgan.losses.modified_discriminator_loss,
        generator_optimizer=tf.train.AdamOptimizer(0.0002, 0.5),
        discriminator_optimizer=tf.train.AdamOptimizer(0.0002, 0.5))

def serving_input_fn():
  inputs = {'inputs': tf.placeholder(tf.float32, [None, 784])}
  return tf.estimator.export.ServingInputReceiver(inputs, inputs)