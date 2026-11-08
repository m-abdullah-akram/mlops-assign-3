"""Task 7 - model-serving REST API (deployed in the cloud; the Flask UI on localhost calls it).

  GET  /health                       -> {"status": "ok", ...}
  POST /predict  (multipart "file")  -> {"label", "confidence", "prob_ai", ...}   needs X-API-Key

Run: flask --app "api.api_server:create_api" run --host 0.0.0.0 --port 8000
"""
import hmac
import logging
import os

from flask import Flask, jsonify, request

from app.validation import ValidationError, validate_upload

log = logging.getLogger("visionguard.api")


def create_api(predictor=None, api_key=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Flask(__name__)
    key = api_key if api_key is not None else os.environ.get("API_KEY", "")
    max_mb = float(os.environ.get("MAX_UPLOAD_MB", "5"))
    app.config["MAX_CONTENT_LENGTH"] = int(max_mb * 1024 * 1024)
    if not key:
        log.warning("API_KEY is empty - every /predict call will be rejected.")
    if predictor is None:
        try:
            from app.inference import Predictor
            predictor = Predictor(os.environ.get("MODEL_PATH", "models/champion/model.keras"),
                                  os.environ.get("THRESHOLD") or None)
        except Exception as exc:  # noqa: BLE001
            log.error("MODEL LOAD FAILED: %s", exc)

    @app.get("/health")
    def health():
        ok = predictor is not None
        return jsonify(status="ok" if ok else "degraded", model_loaded=ok,
                       model_version=getattr(predictor, "model_version", None)), 200 if ok else 503

    @app.post("/predict")
    def predict():
        supplied = request.headers.get("X-API-Key", "")
        if not key or not hmac.compare_digest(supplied.encode(), key.encode()):
            log.warning("Rejected request: bad or missing API key")
            return jsonify(error="invalid or missing API key"), 401
        if predictor is None:
            return jsonify(error="model not loaded"), 503
        try:
            filename, data = validate_upload(request.files.get("file"))
        except ValidationError as exc:
            return jsonify(error=exc.message), exc.status
        result = predictor.predict(data, filename)
        log.info("PREDICT file=%s label=%s confidence=%.4f latency_ms=%s", filename,
                 result["label"], result["confidence"], result["latency_ms"])
        return jsonify(result)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error=f"file larger than {max_mb:g} MB"), 413

    return app
