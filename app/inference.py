"""Task 4.1 - inference module: the model is loaded ONCE at start-up and reused for every request.

Loading per request would cost seconds of disk I/O + graph building each time and would exhaust
memory under concurrency. Preprocessing is the same function used in training
(visionguard.preprocess.decode_resize), so the model sees identical inputs at train and serve time.
"""
import json
import logging
import threading
import time
from pathlib import Path

log = logging.getLogger("visionguard.inference")


class Predictor:
    source = "local-model"

    def __init__(self, model_path, threshold=None, model_name=None, model_version=None):
        import tensorflow as tf
        from visionguard.preprocess import decode_resize

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        meta_file = path.with_name("metadata.json")
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        self.model_name = model_name or meta.get("model_name", path.stem)
        self.model_version = str(model_version or meta.get("model_version", "unknown"))
        self.threshold = float(threshold if threshold is not None else meta.get("threshold", 0.5))
        self._tf, self._decode = tf, decode_resize
        self._model = tf.keras.models.load_model(path, compile=False)
        self.img_size = int(self._model.input_shape[1])
        self._lock = threading.Lock()
        self._model(tf.zeros((1, self.img_size, self.img_size, 3)), training=False)   # warm-up
        log.info("Loaded model %s v%s from %s (threshold=%.3f)", self.model_name,
                 self.model_version, path, self.threshold)

    @property
    def ready(self):
        return True

    def predict(self, image_bytes, filename=""):
        t0 = time.perf_counter()
        img = self._decode(image_bytes, self.img_size)
        with self._lock:
            p_ai = float(self._model(img[None], training=False)[0, 0])
        label = "ai" if p_ai >= self.threshold else "real"
        return {"label": label, "confidence": p_ai if label == "ai" else 1.0 - p_ai,
                "prob_ai": p_ai, "threshold": self.threshold, "model_name": self.model_name,
                "model_version": self.model_version,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
