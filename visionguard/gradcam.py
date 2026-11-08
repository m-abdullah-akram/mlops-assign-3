"""Bonus - Grad-CAM for 4 correct and 4 wrong test predictions (transfer models)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402

from visionguard import config, data, models  # noqa: E402
from visionguard.preprocess import decode_resize  # noqa: E402


def heatmap(model, img):
    """img: float32 (H, W, 3) in [0, 255]. Returns (cam in [0,1], p_ai)."""
    base = models.get_base(model)
    idx = model.layers.index(base)
    before, after = model.layers[1:idx], model.layers[idx + 1:]
    x = tf.convert_to_tensor(img[None])
    with tf.GradientTape() as tape:
        tape.watch(x)
        h = x
        for layer in before:
            h = layer(h)
        conv = base(h, training=False)
        tape.watch(conv)
        out = conv
        for layer in after:
            out = layer(out, training=False) if isinstance(layer, tf.keras.layers.Dropout) else layer(out)
    grads = tape.gradient(out[:, 0], conv)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv, axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam, float(out[0, 0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="outputs/<run> containing model.keras + metrics.json")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    model = tf.keras.models.load_model(run_dir / "model.keras")
    thr = json.loads((run_dir / "metrics.json").read_text())["threshold"]
    size = model.input_shape[1]
    paths, y = data.load_split("test")
    probs = model.predict(data.make_dataset("test", size), verbose=0).ravel()
    pred = (probs >= thr).astype(int)
    right, wrong = np.where(pred == y)[0][:4], np.where(pred != y)[0][:4]
    chosen = [(i, "correct") for i in right] + [(i, "wrong") for i in wrong]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    for ax, (i, kind) in zip(axes.ravel(), chosen):
        img = decode_resize(tf.io.read_file(paths[i]), size).numpy()
        cam, p = heatmap(model, img)
        cam = np.array(Image.fromarray((cam * 255).astype("uint8")).resize((size, size)))
        ax.imshow(img.astype("uint8"))
        ax.imshow(cam, cmap="jet", alpha=0.4)
        ax.set_title(f"{kind}: true={config.CLASS_NAMES[int(y[i])]} p(ai)={p:.2f}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES_DIR / "gradcam.png"
    fig.savefig(out, dpi=120)
    print("Saved", out)


if __name__ == "__main__":
    main()
