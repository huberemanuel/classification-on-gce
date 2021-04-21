import os
import argparse
from datetime import datetime

import tensorflow as tf
from google.cloud import storage
import tensorflow_datasets as tfds
from tensorflow.keras.applications import MobileNetV2


def load_data(batch_size : int = 32) -> tuple:
    train_dataset, test_dataset = tfds.load("mnist", 
                                            split=["train", "test"], 
                                            as_supervised=True)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    size = (32, 32)

    # Resize image, transform to one-hot encoding, convert from grayscale to rgb
    train_dataset = train_dataset.map(lambda x, y: (tf.image.grayscale_to_rgb(tf.image.resize(x, size)), tf.one_hot(y, depth=10)))
    test_dataset = test_dataset.map(lambda x, y: (tf.image.grayscale_to_rgb(tf.image.resize(x, size)), tf.one_hot(y, depth=10)))

    train_dataset = train_dataset.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y))
    test_dataset = test_dataset.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y))

    train_dataset = train_dataset.cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    return train_dataset, test_dataset


def load_model(lr_rate : float = 0.0001):
    """ Loads MobileNetV2 for fine-tuning on MNist dataset """
    base_model = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 3),
                                                include_top=False,
                                                weights='imagenet')

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(10) # 10 classes
    inputs = tf.keras.Input(shape=(32, 32, 3))
    
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    return model

def train(epochs: int = 1):
    train_dataset, test_dataset = load_data()

    model = load_model()

    model.fit(train_dataset, epochs=epochs, verbose=2, validation_data=test_dataset)

    export_model(model)

def export_model(model: tf.keras.Model):

    now = datetime.now()
    model_name = f"digits_model-{now.strftime('%Y-%m-%d-%H-%M-%S')}"
    model.save(model_name)

    # TODO: create constants module
    storage_client = storage.Client("data-lake-test-gcp-iac")
    bucket = storage_client.get_bucket("gcp_iac_bucket_dev")
    dest_file_name = f"models/{model_name}"
    for root, dirs, files in os.walk(model_name):
        for file_name in files:
            ori_file_name = os.path.join(root, file_name)
            blob = bucket.blob(os.path.join("models", ori_file_name))
            blob.upload_from_filename(ori_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a MobileNetV2 model on MNist Data")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs for training")
    args = parser.parse_args()
    train(args.n_epochs)
