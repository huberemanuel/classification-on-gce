import argparse

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import tensorflow_datasets as tfds



def load_data(batch_size : int = 32) -> tuple:
    train_dataset, test_dataset = tfds.load("mnist", 
                                            split=["train", "test"], 
                                            as_supervised=True)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    size = (32, 32)

    # Resize image, transform to one-hot encoding, convert from grayscale to rgb
    train_dataset = train_dataset.map(lambda x, y: (tf.image.grayscale_to_rgb(tf.image.resize(x, size)), tf.one_hot(y, depth=10)))
    test_dataset = test_dataset.map(lambda x, y: (tf.image.grayscale_to_rgb(tf.image.resize(x, size)), tf.one_hot(y, depth=10)))

    train_dataset = train_dataset.cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    return train_dataset, test_dataset


def load_model(lr_rate : float = 0.0001):
    """ Loads MobileNetV2 for fine-tuning on MNist dataset """
    base_model = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 3),
                                                include_top=False,
                                                weights='imagenet')

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    base_model.trainable = False

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
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    return model

def train(epochs : int = 1):
    train_dataset, test_dataset = load_data()
    print(train_dataset, test_dataset)

    model = load_model()

    model.fit(train_dataset, epochs=epochs, verbose=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a MobileNetV2 model on MNist Data")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs for training")
    args = parser.parse_args()
    train(args.n_epochs)