"""Task 8 - predictor that calls the cloud-hosted model API instead of loading a model locally."""
import requests


class RemoteInferenceError(Exception):
    pass


class RemotePredictor:
    source = "remote-api"

    def __init__(self, base_url, api_key, timeout=30):
        self.base_url, self.api_key, self.timeout = base_url.rstrip("/"), api_key, timeout
        self.model_name, self.model_version, self.threshold = "remote", "?", None

    def predict(self, image_bytes, filename="image.jpg"):
        try:
            r = requests.post(f"{self.base_url}/predict", files={"file": (filename, image_bytes)},
                              headers={"X-API-Key": self.api_key}, timeout=self.timeout)
        except requests.RequestException as exc:
            raise RemoteInferenceError(f"Could not reach the model API: {exc.__class__.__name__}")
        if r.status_code == 401:
            raise RemoteInferenceError("The model API rejected the API key (HTTP 401).")
        if r.status_code != 200:
            detail = r.json().get("error", r.text[:120]) if r.headers.get(
                "content-type", "").startswith("application/json") else r.text[:120]
            raise RemoteInferenceError(f"Model API error (HTTP {r.status_code}): {detail}")
        return r.json()

    @property
    def ready(self):
        try:
            return requests.get(f"{self.base_url}/health", timeout=5).status_code == 200
        except requests.RequestException:
            return False
