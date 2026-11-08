"""Task 2.7 toy: random search over learning rate and units on Fashion-MNIST."""
import keras_tuner as kt
import tensorflow as tf

(x, y), (xv, yv) = tf.keras.datasets.fashion_mnist.load_data()
x, xv = x[:10000] / 255.0, xv[:2000] / 255.0
y, yv = y[:10000], yv[:2000]


def build(hp):
    m = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                             tf.keras.layers.Dense(hp.Int("units", 16, 128, step=16), activation="relu"),
                             tf.keras.layers.Dense(10, activation="softmax")])
    m.compile(tf.keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
              "sparse_categorical_crossentropy", ["accuracy"])
    return m


tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=6, seed=42,
                        directory="outputs/tuner_toy", project_name="fashion", overwrite=True)
tuner.search(x, y, epochs=3, validation_data=(xv, yv))
tuner.results_summary()
print("Best:", tuner.get_best_hyperparameters(1)[0].values)
