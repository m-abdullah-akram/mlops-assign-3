"""Task 3.1 / 3.2 toy: tiny Keras model on MNIST, logged to MLflow.

Local:    python labs/mlflow_toy.py --lr 0.01 && python labs/mlflow_toy.py --lr 0.001 && mlflow ui
DagsHub:  export the MLFLOW_TRACKING_* variables from .env first, then run the same command.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import tensorflow as tf  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()
ap = argparse.ArgumentParser()
ap.add_argument("--lr", type=float, default=0.01)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--local", action="store_true", help="ignore .env and use ./mlruns")
a = ap.parse_args()

if a.local or not os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("toy-mnist")
tf.keras.utils.set_random_seed(42)
(x, y), (xt, yt) = tf.keras.datasets.mnist.load_data()
x, xt = x[:10000] / 255.0, xt[:2000] / 255.0
y = y[:10000]
yt = yt[:2000]

model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                             tf.keras.layers.Dense(64, activation="relu"),
                             tf.keras.layers.Dense(10, activation="softmax")])
model.compile(tf.keras.optimizers.Adam(a.lr), "sparse_categorical_crossentropy", ["accuracy"])

with mlflow.start_run(run_name=f"toy-lr{a.lr}"):
    mlflow.set_tag("purpose", "tool-lab")
    mlflow.log_params({"lr": a.lr, "epochs": a.epochs, "units": 64})
    hist = model.fit(x, y, epochs=a.epochs, validation_data=(xt, yt), verbose=2)
    for e in range(a.epochs):
        mlflow.log_metrics({k: v[e] for k, v in hist.history.items()}, step=e)
    fig, ax = plt.subplots()
    ax.plot(hist.history["loss"], label="train")
    ax.plot(hist.history["val_loss"], label="val")
    ax.legend()
    ax.set_title(f"lr={a.lr}")
    mlflow.log_figure(fig, "loss_curve.png")
    mlflow.tensorflow.log_model(model, artifact_path="model")
    print("Logged to", mlflow.get_tracking_uri())
