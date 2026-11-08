"""Task 2.5 / 2.6 - threshold choice, metrics, confusion matrix, ROC, error analysis."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, fbeta_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score, roc_curve)

from visionguard import config  # noqa: E402


def choose_threshold(y_val, p_val, beta=1.0):
    """Pick the threshold that maximises F-beta on the VALIDATION set (never on test).

    beta=1 weighs both error types equally; beta>1 penalises missed AI images (false negatives).
    """
    grid = np.linspace(0.05, 0.95, 91)
    scores = [fbeta_score(y_val, p_val >= t, beta=beta, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def compute_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def plot_confusion(y, p, threshold, path, title="Confusion matrix (test)"):
    cm = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], config.CLASS_NAMES)
    ax.set_yticks([0, 1], config.CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title}\nthreshold={threshold:.2f}")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_roc(y, p, path, label="model"):
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate (real called AI)")
    ax.set_ylabel("True positive rate (AI caught)")
    ax.set_title("ROC curve (test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_curves(histories, best_epochs, path, title):
    """histories: list of Keras history dicts (one per stage); best_epochs: 1-based per stage."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    offset = 0
    for hist, best in zip(histories, best_epochs):
        n = len(hist["loss"])
        xs = np.arange(1, n + 1) + offset
        for ax, key in zip(axes, ["loss", "accuracy"]):
            ax.plot(xs, hist[key], label=f"train {key}")
            ax.plot(xs, hist[f"val_{key}"], label=f"val {key}")
            ax.axvline(best + offset, color="red", ls="--", lw=1)
        offset += n
    for ax, key in zip(axes, ["Loss", "Accuracy"]):
        ax.set_xlabel("epoch (stages concatenated)")
        ax.set_title(f"{title}: {key} (red = best epoch of each stage)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def misclassified_grid(paths, y, p, threshold, path, n=10):
    """Show the n most confidently wrong test images (fewer if the model makes fewer errors)."""
    pred = (p >= threshold).astype(int)
    wrong = np.where(pred != y)[0]
    wrong = wrong[np.argsort(-np.abs(p[wrong] - threshold))][:n]
    cols = 5
    rows = max(1, int(np.ceil(len(wrong) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.3 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, idx in zip(np.atleast_1d(axes).ravel(), wrong):
        ax.imshow(Image.open(paths[idx]).convert("RGB"))
        ax.set_title(f"true={config.CLASS_NAMES[int(y[idx])]} pred={config.CLASS_NAMES[pred[idx]]}\n"
                     f"p(ai)={p[idx]:.2f}  #{idx}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return [int(i) for i in wrong]
