"""Task 1.3 - sample grids and histograms per class, built from reports/audit.csv."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from visionguard import config  # noqa: E402


def sample_grid(df, per_class, out):
    classes = sorted(df["class_name"].unique())
    fig, axes = plt.subplots(len(classes), per_class, figsize=(2.6 * per_class, 2.8 * len(classes)))
    axes = axes.reshape(len(classes), per_class)
    for r, cls in enumerate(classes):
        sub = df[df["class_name"] == cls].sample(min(per_class, (df["class_name"] == cls).sum()),
                                                 random_state=config.SEED)
        for c in range(per_class):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(sub):
                ax.imshow(Image.open(sub.iloc[c]["path"]).convert("RGB"))
                if c == 0:
                    ax.set_title(cls, loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def histograms(df, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, title in zip(axes, ["width", "height", "file_size"],
                              ["Width (px)", "Height (px)", "File size (bytes)"]):
        for cls, sub in df.groupby("class_name"):
            ax.hist(sub[col], bins=30, alpha=0.55, label=cls)
        ax.set_title(title)
        ax.set_ylabel("images")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", default=str(config.REPORTS_DIR / "audit.csv"))
    ap.add_argument("--per-class", type=int, default=6)
    args = ap.parse_args()
    df = pd.read_csv(args.audit)
    df = df[~df["corrupt"]]
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sample_grid(df, args.per_class, config.FIGURES_DIR / "sample_grid.png")
    histograms(df, config.FIGURES_DIR / "histograms.png")
    print("Saved figures to", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
