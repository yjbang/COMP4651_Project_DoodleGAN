from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tqdm import tqdm

import argparse
import os
import random
import sys
import numpy as np
import tensorflow as tf

def parse_npy(file_,observations_per_class):
    """Parse an ndjson line and return doodle_flat (as np array) and classname."""

    data = np.load(file_)
    length = len(data)
    reading_order = random.sample(range(0, length-1), observations_per_class)
    doodle_flats = [data[i] for i in reading_order]

    return doodle_flats



def convert_data(trainingdata_dir,
                 observations_per_class,
                 output_file,
                 classnames,
                 output_shards=10,
                 offset=0):
  """Convert training data from ndjson files into tf.Example in tf.Record.

  Args:
   trainingdata_dir: path to the directory containin the training data.
   observations_per_class: the number of items to load per class.
   output_file: path where to write the output.
   classnames: array with classnames - is auto created if not passed in.
   output_shards: the number of shards to write the output in.

  Returns:
    classnames: the class names as strings. classnames[classes[i]] is the
      textual representation of the class of the i-th data point.
  """

  def _pick_output_shard():
    return random.randint(0, output_shards - 1)

  writers = []
  for i in range(FLAGS.output_shards):
    writers.append(
        tf.python_io.TFRecordWriter("%s-%05i-of-%05i" % (output_file, i,
                                                         output_shards)))

  for filename in tqdm(sorted(tf.gfile.ListDirectory(trainingdata_dir))):
    if not filename.endswith(".npy"):
      print("Skipping", filename)
      continue
    file_ = trainingdata_dir+'/'+filename
    doodle_flats = parse_npy(file_, observations_per_class)
    class_name = filename[:-4].replace(" ", "_")

    if class_name not in classnames:
        classnames.append(class_name)

    for doodle_flat in doodle_flats:
        features = {}
        features["class_index"] = tf.train.Feature(int64_list=tf.train.Int64List(
            value=[classnames.index(class_name)]))
        features["doodle"] = tf.train.Feature(int64_list=tf.train.Int64List(
            value=doodle_flat))

        f = tf.train.Features(feature=features)
        example = tf.train.Example(features=f)
        writers[_pick_output_shard()].write(example.SerializeToString())


  # Close all files
  for w in writers:
    w.close()
  # Write the class list.
  with tf.gfile.GFile(output_file + ".classes", "w") as f:
    for class_name in classnames:
      f.write(class_name + "\n")
  return classnames


def main(argv):
  del argv
  classnames = convert_data(
      FLAGS.npy_path,
      FLAGS.train_observations_per_class,
      os.path.join(FLAGS.output_path, "training.tfrecord"),
      classnames=[],
      output_shards=FLAGS.output_shards,
      offset=0)
  # convert_data(
  #     FLAGS.ndjson_path,
  #     FLAGS.eval_observations_per_class,
  #     os.path.join(FLAGS.output_path, "eval.tfrecord"),
  #     classnames=classnames,
  #     output_shards=FLAGS.output_shards,
  #     offset=FLAGS.train_observations_per_class)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--npy_path",
      type=str,
      default="",
      help="Directory where the ndjson files are stored.")
  parser.add_argument(
      "--output_path",
      type=str,
      default="",
      help="Directory where to store the output TFRecord files.")
  parser.add_argument(
      "--train_observations_per_class",
      type=int,
      default=10000,
      help="How many items per class to load for training.")
  # parser.add_argument(
  #     "--eval_observations_per_class",
  #     type=int,
  #     default=1000,
  #     help="How many items per class to load for evaluation.")
  parser.add_argument(
      "--output_shards",
      type=int,
      default=10,
      help="Number of shards for the output.")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
