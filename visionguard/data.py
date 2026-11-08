"""Task 2.1 - tf.data input pipeline driven by the split CSVs."""
import pandas as pd
import tensorflow as tf

from visionguard import config
from visionguard.preprocess import decode_resize

AUTOTUNE = tf.data.AUTOTUNE


def load_split(name):
    df = pd.read_csv(config.SPLIT_DIR / f"{name}.csv")
    paths = [str(config.ROOT / p) for p in df["path"]]
    return paths, df["label"].to_numpy().astype("float32")


def make_augmenter():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.15),
    ], name="augment")


def make_dataset(split, img_size=config.IMG_SIZE, batch_size=32, training=False,
                 augment=True, seed=config.SEED):
    """read path -> decode -> resize -> (shuffle) -> batch -> (augment, train only) -> prefetch."""
    paths, labels = load_split(split)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (decode_resize(tf.io.read_file(p), img_size), y),
                num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    if training and augment:
        aug = make_augmenter()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def save_augmented_batch_figure(out_path, img_size=config.IMG_SIZE, n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = make_dataset("train", img_size, batch_size=n, training=True, augment=True)
    x, y = next(iter(ds))
    fig, axes = plt.subplots(2, n // 2, figsize=(3 * n // 2, 6))
    for ax, img, lab in zip(axes.ravel(), x.numpy(), y.numpy()):
        ax.imshow(img.clip(0, 255).astype("uint8"))
        ax.set_title(config.CLASS_NAMES[int(lab)])
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_augmented_batch_figure(config.FIGURES_DIR / "augmented_batch.png")
    print("Saved", config.FIGURES_DIR / "augmented_batch.png")
