import io

from PIL import Image


def make_image_bytes(fmt="PNG", size=(64, 48), color=(120, 30, 200)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, fmt)
    return buf.getvalue()


class FakePredictor:
    """Stands in for the TensorFlow model so tests run without TensorFlow."""
    source = "fake-model"
    ready = True
    model_name, model_version, threshold = "fake_model", "7", 0.5

    def predict(self, image_bytes, filename=""):
        return {"label": "ai", "confidence": 0.9, "prob_ai": 0.9, "threshold": 0.5,
                "model_name": "fake_model", "model_version": "7", "latency_ms": 1.0}
