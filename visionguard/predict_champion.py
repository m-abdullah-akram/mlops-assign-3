"""Task 3.6 - separate script: load the model BY ALIAS from the registry and classify one image."""
import argparse
import tensorflow as tf

from visionguard import config, registry
from visionguard.preprocess import decode_resize

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--name", default=f"{config.PROJECT_NAME}_model")
ap.add_argument("--alias", default="champion")
a = ap.parse_args()

model, version, threshold, run_id = registry.load_champion(a.name, a.alias)
size = model.input_shape[1]
img = decode_resize(tf.io.read_file(a.image), size)
p_ai = float(model(img[None], training=False)[0, 0])
label = "ai" if p_ai >= threshold else "real"
conf = p_ai if label == "ai" else 1 - p_ai
print(f"model={a.name} version={version} alias={a.alias} run_id={run_id}")
print(f"image={a.image} -> {label.upper()} (confidence {conf:.1%}, p(ai)={p_ai:.4f}, threshold={threshold})")
