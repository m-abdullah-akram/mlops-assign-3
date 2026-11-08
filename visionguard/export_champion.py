"""Download the champion ONCE into models/champion/ so the server never calls DagsHub per request.

Run this on your machine before `docker compose up` (the folder is mounted into the container).
"""
import argparse
import json

from visionguard import config, registry

ap = argparse.ArgumentParser()
ap.add_argument("--name", default=f"{config.PROJECT_NAME}_model")
ap.add_argument("--alias", default="champion")
a = ap.parse_args()

model, version, threshold, run_id = registry.load_champion(a.name, a.alias)
out = config.MODELS_DIR / "champion"
out.mkdir(parents=True, exist_ok=True)
model.save(out / "model.keras")
(out / "metadata.json").write_text(json.dumps({
    "model_name": a.name, "model_version": str(version), "threshold": threshold,
    "run_id": run_id, "alias": a.alias}, indent=2))
print("Exported to", out, "| version", version, "| threshold", threshold)
