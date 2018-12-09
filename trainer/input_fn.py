import tensorflow as tf
import functools


def input_fn(filenames, config=None):
    """Estimator `input_fn`.
    """
    # Preprocess 10 files concurrently and interleaves records from each file.
    dataset = tf.data.TFRecordDataset.list_files(filenames)
    dataset = dataset.shuffle(buffer_size=10)
    dataset = dataset.repeat()

    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=10,
        block_length=1)

    dataset = (dataset
               .map(functools.partial(parse_fn), num_parallel_calls=4)
               .shuffle(buffer_size=1000000)
               .repeat()
               .batch(config.batch_size)
               # .prefetch(config.batch_size)
               )
    features, labels = dataset.make_one_shot_iterator().get_next()
    print("input dimensions: ", features.shape, labels.shape)

    return (features, labels)


def parse_fn(drawit_proto):
    """Parse a single record which is expected to be a tensorflow.Example."""
    num_classes = 345

    features = {"doodle": tf.FixedLenFeature((28 * 28), dtype=tf.int64),
                "class_index": tf.FixedLenFeature((), tf.int64, default_value=0)}

    parsed_features = tf.parse_single_example(drawit_proto, features)

    labels = parsed_features["class_index"]
    labels = tf.one_hot(labels, num_classes)

    features = parsed_features['doodle']

    features = tf.reshape(features, [28, 28, 1])
    features = tf.cast(features, tf.float32)

    features = (features / 127.5) - 1

    return features, labels


def mnist_input(config):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    def train_preprocess(image, label):
        image = (image / 127.5) - 1
        image = tf.reshape(image, [28, 28, 1])
        label = tf.one_hot(label, config.num_classes)
        return image, label

    def create_mnist_dataset(data, labels, batch_size):
        def gen():
            for image, label in zip(data, labels):
                yield image, label

        ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32), ((28, 28), ()))

        return ds.map(train_preprocess).shuffle(len(data)).repeat().batch(batch_size)

    # train and validation dataset with different batch size
    train_dataset = create_mnist_dataset(x_train, y_train, config.batch_size)
    # valid_dataset = create_mnist_dataset(x_test, y_test, config.batch_size)

    image, label = train_dataset.make_one_shot_iterator().get_next()
    print("input dimensions: ", image.shape, label.shape)

    return (image, label)
