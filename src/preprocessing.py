import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def split_data(development_df, test_size=0.20):
    train_df, val_df = train_test_split(
        development_df,
        test_size=test_size,
        random_state=SEED,
        stratify=development_df["label"]
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True)
    )


def load_and_preprocess_image(filepath, label):
    image = tf.io.read_file(filepath)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, tf.cast(label, tf.float32)


def create_tf_dataset(dataframe, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((
        dataframe["filepath"].values,
        dataframe["label"].values.astype(np.float32)
    ))

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED
        )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def calculate_class_weights(train_df):
    classes = np.array([0, 1])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"].values
    )

    return {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, weights)
    }