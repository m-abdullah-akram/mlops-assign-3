"""VisionGuard Flask web app (Tasks 4, 5, 8).

Run:  flask --app "app.app:create_app" run --host 0.0.0.0 --port 5000     (debug stays OFF)
"""
import base64
import logging
import os
import time

from flask import Flask, jsonify, render_template, request

from app.db import Database
from app.remote import RemoteInferenceError, RemotePredictor
from app.validation import ValidationError, validate_upload

log = logging.getLogger("visionguard.app")

CAVEAT = ("This is a statistical estimate, not proof. Edited, heavily compressed, stylised or "
          "very small images can fool the model in both directions.")


def load_settings():
    e = os.environ.get
    return {
        "MODEL_PATH": e("MODEL_PATH", "models/champion/model.keras"),
        "THRESHOLD": e("THRESHOLD") or None,
        "DATABASE_URL": e("DATABASE_URL", "sqlite:///instance/visionguard.db"),
        "MAX_UPLOAD_MB": float(e("MAX_UPLOAD_MB", "5")),
        "INFERENCE_MODE": e("INFERENCE_MODE", "local"),          # local | remote
        "REMOTE_API_URL": e("REMOTE_API_URL", ""),
        "REMOTE_API_KEY": e("REMOTE_API_KEY", ""),
    }


def create_app(overrides=None, predictor=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Flask(__name__)
    cfg = {**load_settings(), **(overrides or {})}
    app.config.update(cfg)
    app.config["MAX_CONTENT_LENGTH"] = int(cfg["MAX_UPLOAD_MB"] * 1024 * 1024)

    db = Database(cfg["DATABASE_URL"])
    db.init()                                   # fails loudly (in the logs) if the DB is unreachable
    app.extensions["db"] = db

    if predictor is None:
        if cfg["INFERENCE_MODE"] == "remote":
            predictor = RemotePredictor(cfg["REMOTE_API_URL"], cfg["REMOTE_API_KEY"])
            log.info("Remote inference via %s", cfg["REMOTE_API_URL"])
        else:
            try:
                from app.inference import Predictor
                predictor = Predictor(cfg["MODEL_PATH"], cfg["THRESHOLD"])
            except Exception as exc:  # noqa: BLE001
                log.error("MODEL LOAD FAILED: %s (path=%s)", exc, cfg["MODEL_PATH"])
    app.extensions["predictor"] = predictor

    @app.get("/")
    def index():
        return render_template("index.html", max_mb=cfg["MAX_UPLOAD_MB"])

    @app.post("/predict")
    def predict():
        pred = app.extensions["predictor"]
        if pred is None:
            return render_template("error.html", title="Model unavailable",
                                   message="The model is not loaded. Check the server logs."), 503
        try:
            filename, data = validate_upload(request.files.get("file"))
        except ValidationError as exc:
            log.warning("Rejected upload: %s", exc.message)
            return render_template("error.html", title="Upload rejected", message=exc.message), exc.status
        try:
            t0 = time.perf_counter()
            result = pred.predict(data, filename)
            result.setdefault("latency_ms", round((time.perf_counter() - t0) * 1000, 1))
        except RemoteInferenceError as exc:
            log.error("Remote inference failed: %s", exc)
            return render_template("error.html", title="Model API problem", message=str(exc)), 502
        db.add(filename=filename, label=result["label"], confidence=result["confidence"],
               prob_ai=result["prob_ai"], threshold=result["threshold"],
               model_name=result["model_name"], model_version=result["model_version"],
               source=pred.source, latency_ms=result["latency_ms"])
        preview = "data:image/jpeg;base64," + base64.b64encode(data).decode()   # never stored on disk
        return render_template("result.html", r=result, filename=filename, caveat=CAVEAT,
                               preview=preview)

    @app.get("/history")
    def history():
        return render_template("history.html", rows=db.recent())

    @app.get("/health")
    def health():
        pred = app.extensions["predictor"]
        db_ok = db.ping()
        model_ok = bool(pred and pred.ready)
        body = {"status": "ok" if (db_ok and model_ok) else "degraded", "database": db_ok,
                "model_loaded": model_ok, "mode": cfg["INFERENCE_MODE"],
                "model_version": getattr(pred, "model_version", None)}
        return jsonify(body), 200 if body["status"] == "ok" else 503

    @app.errorhandler(413)
    def too_large(_):
        return render_template("error.html", title="File too large",
                               message=f"Images must be {cfg['MAX_UPLOAD_MB']:g} MB or smaller."), 413

    @app.errorhandler(404)
    def not_found(_):
        return render_template("error.html", title="Page not found",
                               message="That page does not exist."), 404

    @app.errorhandler(500)
    def server_error(_):
        return render_template("error.html", title="Something went wrong",
                               message="An unexpected error occurred. It has been logged."), 500

    return app
