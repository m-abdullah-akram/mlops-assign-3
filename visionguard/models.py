"""Task 2.2 / 2.3 - baseline CNN and transfer-learning model builders."""
import tensorflow as tf
from tensorflow.keras import layers

from visionguard import config

# rescale = (scale, offset) reproduces the preprocessing the backbone was trained with.
# MobileNetV2 expects [-1, 1]  ->  x / 127.5 - 1.   EfficientNetB0 rescales internally (0-255 in).
BACKBONES = {
    "mobilenetv2": {"cls": tf.keras.applications.MobileNetV2, "rescale": (1 / 127.5, -1.0),
                    "block_start": "block_16_expand"},
    "efficientnetb0": {"cls": tf.keras.applications.EfficientNetB0, "rescale": None,
                       "block_start": "block7a_expand_conv"},
}


def build_baseline(img_size=config.IMG_SIZE, dropout=0.4):
    inputs = tf.keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1 / 255.0, name="rescale")(inputs)
    for i, filters in enumerate([32, 64, 128, 256], start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="ai_probability")(x)
    return tf.keras.Model(inputs, outputs, name="baseline_cnn")


def build_transfer(backbone="mobilenetv2", dropout=0.3, img_size=config.IMG_SIZE):
    spec = BACKBONES[backbone]
    inputs = tf.keras.Input((img_size, img_size, 3), name="image")
    x = inputs
    if spec["rescale"]:
        x = layers.Rescaling(*spec["rescale"], name="preprocess")(x)
    base = spec["cls"](include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
    base.trainable = False                       # stage 1: frozen feature extractor
    x = base(x, training=False)                  # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="ai_probability")(x)
    return tf.keras.Model(inputs, outputs, name=f"visionguard_{backbone}")


def get_base(model):
    """Return the nested pretrained backbone of a transfer model (None for the baseline)."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    return None


def unfreeze(model, backbone, n_last=None):
    """Unfreeze the last block (default) or the last `n_last` layers; BatchNorm stays frozen."""
    base = get_base(model)
    if n_last == 0:
        base.trainable = False
        return
    names = [layer.name for layer in base.layers]
    start = names.index(BACKBONES[backbone]["block_start"]) if n_last is None \
        else max(0, len(names) - n_last)
    base.trainable = True
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= start and not isinstance(layer, layers.BatchNormalization)


def trainable_layer_names(model):
    base = get_base(model)
    names = [layer.name for layer in model.layers if layer is not base and layer.trainable_weights]
    if base is not None:
        names += [f"{base.name}/{layer.name}" for layer in base.layers if layer.trainable_weights]
    return names


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model
