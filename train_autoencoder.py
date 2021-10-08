from tensorflow import keras
from tensorflow.keras import layers

filename = '/external1/gqa_dataset/code/sub-CSI1.tfrecords'
dataset = tf.data.TFRecordDataset(filenames = [filename]).shuffle(10000).batch(15)
def tf_parse(eg):
    example = tf.io.parse_example(
        eg, {
            'x': tf.io.FixedLenFeature(shape=(71, 89, 72), dtype=tf.float32),
            'y': tf.io.FixedLenFeature(shape=(71, 89, 72), dtype=tf.float32),
        })
    return example['x'], example['y']
decoded = dataset.map(tf_parse)
print(decoded)

for x,y in decoded:
    print(x.shape)

encoder_input = keras.Input(shape=(71, 89, 72), name="img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(16, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.Flatten()(x)
print(encoder_output.shape)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

x = layers.Reshape((18, 24, 16))(encoder_output)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(72, 4, activation="relu")(x)

autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")

autoencoder.summary()

autoencoder.compile(
    loss='mse',
    #optimizer=keras.optimizers.RMSprop(),
    optimizer='adam',
    metrics=[tf.keras.metrics.MeanSquaredError()],
)
autoencoder.fit(decoded, epochs=5)
