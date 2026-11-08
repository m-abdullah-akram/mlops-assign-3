"""MLflow / DagsHub helpers. Credentials come ONLY from environment variables (.env)."""
import re
import subprocess

import mlflow
import tensorflow as tf

from visionguard import config


def setup_mlflow(experiment=None):
    """Uses MLFLOW_TRACKING_URI / MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD if set,
    otherwise falls back to a local ./mlruns store (Task 3.1)."""
    import os
    uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(uri if uri else f"file:{config.ROOT / 'mlruns'}")
    mlflow.set_experiment(experiment or config.PROJECT_NAME)
    return mlflow.get_tracking_uri()


def _run(cmd):
    try:
        return subprocess.check_output(cmd, cwd=config.ROOT, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def git_commit():
    return _run(["git", "rev-parse", "HEAD"])


def dvc_data_version():
    """md5 recorded in the DVC pointer file of the dataset (data/raw.dvc)."""
    pointer = config.ROOT / "data" / "raw.dvc"
    if pointer.exists():
        m = re.search(r"md5:\s*(\S+)", pointer.read_text())
        if m:
            return m.group(1)
    return "untracked"


def standard_tags(seed):
    return {"git_commit": git_commit(), "dvc_data_version": dvc_data_version(),
            "seed": str(seed), "project": config.PROJECT_NAME}


class MlflowEpochCallback(tf.keras.callbacks.Callback):
    """Manual logging: per-epoch metrics for every stage on one continuous step axis."""

    def __init__(self, stage, step_offset=0):
        super().__init__()
        self.stage, self.offset = stage, step_offset

    def on_epoch_end(self, epoch, logs=None):
        metrics = {k: float(v) for k, v in (logs or {}).items()}
        metrics["stage"] = float(self.stage)
        try:
            metrics["lr"] = float(self.model.optimizer.learning_rate.numpy())
        except Exception:  # noqa: BLE001
            pass
        mlflow.log_metrics(metrics, step=self.offset + epoch)
