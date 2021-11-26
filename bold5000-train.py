import tensorflow as tf
import os
import time
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
data_dir = '/local-scratch/localhome/mkhademi/BOLD5000_2.0/'
dataset = tf.data.TFRecordDataset(filenames = [data_dir + 'image_data/bold5000_imagenet.tfrecords']).batch(15)
# dataset = tf.data.TFRecordDataset(filenames = [data_dir + 'image_data/bold5000_imagenet.tfrecords']).shuffle(1916).batch(15)
train_ds = dataset.take(1000)
# test_ds = dataset.skip(1800)
test_ds = dataset.take(816)

def tf_parse(eg):
    example = tf.io.parse_example(
        eg, {
            'x': tf.io.FixedLenFeature(shape=(71, 89, 72), dtype=tf.float32),
            'y': tf.io.FixedLenFeature(shape=(1000), dtype=tf.float32),
            'y_coco': tf.io.FixedLenFeature(shape=(90), dtype=tf.int64),
            'y_imagenet': tf.io.FixedLenFeature(shape=(1000), dtype=tf.int64),
        })
    return example['x'], example['y_imagenet']
    
decoded = train_ds.map(tf_parse)
decoded_test = test_ds.map(tf_parse)
 
raw_example = next(iter(test_ds))
# print(tf_parse(raw_example)[0].numpy().shape)
# print(tf_parse(raw_example)[1].numpy().shape)
# print(tf_parse(raw_example)[1].numpy().sum(axis=1))

model = tf.keras.Sequential([
    tf.keras.Input(shape=(71, 89, 72)),
    tf.keras.layers.Conv2D(4, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(8, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(rate=0.2),
    # tf.keras.layers.Dense(1000, activation='sigmoid'),
    tf.keras.layers.Dense(1000),
    # tf.keras.layers.Softmax()
    ])
    
model.summary()
model.compile(
    # loss='mse',
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    # optimizer=keras.optimizers.RMSprop(),
    optimizer='adam',
    # metrics=[tf.keras.metrics.MeanSquaredError()],
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
    # metrics=[tf.keras.metrics.CategoricalCrossentropy(from_logits=True)]
)
model.fit(decoded, epochs=30)

loss, accuracy = model.evaluate(decoded_test)
print("Loss :", loss)
print("Accuracy :", accuracy)
