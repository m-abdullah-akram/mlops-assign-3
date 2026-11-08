"""The ONE preprocessing function used by training (tf.data) and serving (Flask/API).

Output is float32 in [0, 255]. Model-specific scaling (e.g. MobileNetV2 -> [-1, 1]) lives
*inside* the Keras model as a Rescaling layer, so it travels with the saved model.
"""
import tensorflow as tf


def decode_resize(image_bytes, size):
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    # Plain resize: the image is stretched to size x size (aspect ratio is NOT preserved).
    img = tf.image.resize(img, (size, size), method="bilinear")
    return tf.cast(img, tf.float32)
