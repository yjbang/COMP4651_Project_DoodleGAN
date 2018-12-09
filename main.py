import os
import logging
import tensorflow as tf
import numpy as np
import skimage.io
from shutil import copy

from trainer.input_fn import input_fn, mnist_input
from trainer.models import GAN
from trainer.trainer import Trainer
from utils import *


# Local Training
def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.sample_dir, config.checkpoint_dir])
    copy(args.config, config.model_dir)

    # create tensorflow session
    sess = tf.Session()

    is_training = True

    if args.data == 'drawit':
        config.num_classes = 345
        # filenames should be type of file pattern
        filenames = 'training_data/training.tfrecord-?????-of-?????'

        # create your data input pipeline
        # in the features - class_index & doodle
        input = input_fn(filenames, config)
        print("using drawit data")

    elif args.data == 'mnist':
        config.num_classes = 10
        input = mnist_input(config)
        print("using mnist data")

    else:
        raise ("no dataset chosen")

    # img, label = sess.run(input)
    #
    # image = denorm(np.squeeze(img[0]))
    # sample_path = os.path.join('test.jpg')
    # skimage.io.imsave(sample_path, image)

    # create instance of the model you want
    model = GAN(config, input)

    # create tensorboard & terminal logger
    logger = Logger(sess, config)
    logger.set_logger(log_path=os.path.join(
        config.model_dir, args.mode + '.log'))

    # enter training or testing mode
    if is_training:
        logging.info(config.exp_description)
        logging.info("creating trainer...")

        # create trainer and path all previous components to it
        trainer = Trainer(sess, model, config, logger)

        # here you train your model
        trainer.train()

    else:
        # load latest checkpoint
        model.load(sess)


if __name__ == '__main__':
    main()
