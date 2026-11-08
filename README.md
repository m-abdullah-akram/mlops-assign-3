# VisionGuard (22l-6632_A2_Ops)

[![CI](https://github.com/mabdullah-akram/22l-6632_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/mabdullah-akram/22l-6632_A2_Ops/actions/workflows/ci.yml)

AI-generated vs real image classifier: TensorFlow/Keras models, MLflow tracking on DagsHub, DVC data
versioning, a Flask web app with a request-history database, Docker Compose, CI, and a cloud model API.

- GitHub: https://github.com/mabdullah-akram/22l-6632_A2_Ops
- DagsHub: https://dagshub.com/ma92365/22l-6632_A2_Ops
- Live model API: `http://YOUR_SERVER_IP:8000` (see `docs/DEPLOYMENT.md`)

## Layout
```
visionguard/   audit, eda, splits, data (tf.data), models, train, tuner, gradcam, registry, export
app/           Flask UI: inference module, validation, db, remote client, templates, CSS
api/           Cloud model-serving REST API (X-API-Key protected)
tests/         unit tests (fake model - no TensorFlow needed)
labs/          toy exercises: DVC, MLflow, Keras Tuner, Docker, GitHub Actions
docs/          DEPLOYMENT.md, SCREENSHOT_CHECKLIST.md, REPORT_TEMPLATE.md
```
Label convention: `0 = real`, `1 = ai` (the "positive" class is AI-generated).

## 0. One-time setup
```bash
python scripts/set_roll_number.py <ROLL_NUMBER> <GITHUB_USER> <DAGSHUB_USER>   # replaces placeholders
python3.11 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                      # then edit; .env is git-ignored
# Task 1.1 evidence:
python -c "import tensorflow as tf, sys; print(sys.version); print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
pip freeze | grep -i -E "tensorflow|mlflow|keras|numpy|pandas|scikit|flask|dvc"
```
Use Python 3.10 or 3.11 (TensorFlow 2.15). Put real values in `.env` (DagsHub token, passwords).

## 1. Data (Task 1)
```bash
# Download from Kaggle (needs ~/.kaggle/kaggle.json) or download the zip manually into data/raw/
kaggle datasets download -d cashbowman/ai-generated-images-vs-real-images -p data/raw --unzip
python -m visionguard.audit      # -> reports/audit.csv + reports/audit_summary.txt (1.2)
python -m visionguard.eda        # -> reports/figures/sample_grid.png, histograms.png (1.3)
python -m visionguard.splits     # -> data/splits/{train,val,test}.csv (dedup, stratified 70/15/15)
python -m visionguard.data       # -> reports/figures/augmented_batch.png (2.1)
```
The audit assumes the Kaggle folders `AiArtData` and `RealArt` (mapped in `visionguard/config.py`
`FOLDER_TO_CLASS`). **Check the per-class counts it prints**; if your folders are named differently, add
them to that dictionary.

DVC (1.4 apply part), after `git init` and creating the DagsHub repo:
```bash
dvc init
dvc add data/raw && git add data/raw.dvc data/.gitignore && git commit -m "Track dataset with DVC"
dvc remote add -d origin https://dagshub.com/ma92365/22l-6632_A2_Ops.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user ma92365
dvc remote modify origin --local password <DAGSHUB_TOKEN>      # stored in .dvc/config.local (git-ignored)
dvc push
git add data/splits reports .dvc && git commit -m "Splits, audit, DVC config" && git push
```
(Check the exact remote URL in DagsHub: *Remote > Data > DVC*.) Toy part: `labs/dvc_toy.sh`.

## 2. Training and tracking (Tasks 2 and 3)
Toy labs first: `python labs/keras_tuner_toy.py`, `python labs/mlflow_toy.py --lr 0.01 --local`
(then `--lr 0.001 --local`, then `mlflow ui`); then without `--local` for DagsHub.

At least 6 runs, each changing ONE thing relative to the reference run (`mnv2-ref`):
```bash
python -m visionguard.train --model baseline --run-name baseline-cnn
python -m visionguard.train --model transfer --run-name mnv2-ref                       # manual logging
python -m visionguard.train --model transfer --run-name mnv2-autolog --tracking autolog # 3.3 autolog
python -m visionguard.train --model transfer --run-name mnv2-noaug --no-augment
python -m visionguard.train --model transfer --run-name mnv2-lr3e-4 --lr 3e-4
python -m visionguard.train --model transfer --run-name mnv2-img160 --img-size 160
python -m visionguard.train --model transfer --run-name effb0 --backbone efficientnetb0
python -m visionguard.tuner --trials 8                  # 2.7 -> reports/tuner_results.json
python -m visionguard.gradcam --run-dir outputs/mnv2-ref   # bonus
```
Each run writes `outputs/<run>/` (curves, confusion matrix, ROC, misclassified grid, summary,
trainable-layer lists, metrics.json) and uploads them to MLflow. Use `--beta 2` if you decide that
calling a fake image real is the worse error (threshold then favours recall of the AI class);
the report text must match the value you used.

Registry (3.6): re-run your best configuration twice with `--register` (two versions), then
```bash
python -m visionguard.set_champion --version 2
python -m visionguard.predict_champion path/to/image.jpg   # loads models:/<name>@champion
python -m visionguard.export_champion                       # copies champion to models/champion/
```
`export_champion` is how the model reaches the server: it is downloaded once and mounted, so the app
never calls DagsHub per request.

## 3. Run the web app (Tasks 4, 5)
```bash
# local, SQLite history, model from models/champion/
flask --app "app.app:create_app" run --port 5000          # debug stays off
# whole stack (Flask + PostgreSQL) in Docker
cp .env.example .env && docker compose up --build -d
docker compose ps            # both services should be "healthy"
curl http://localhost:5000/health
```
Pages: `/` upload, `/predict` result, `/history`, `/health`.

## 4. Tests and CI
```bash
pip install -r requirements-ci.txt && python -m pytest -q && flake8 visionguard app api tests
```

## 5. Cloud API and integration (Tasks 7 and 8)
See `docs/DEPLOYMENT.md`. In short: deploy `docker-compose.api.yml` on a VM; run the local Flask app with
`INFERENCE_MODE=remote REMOTE_API_URL=http://<ip>:8000 REMOTE_API_KEY=<key>`.

## Report
Fill in `docs/REPORT_TEMPLATE.md` following `docs/SCREENSHOT_CHECKLIST.md`, export to PDF and name it
`<RollNumber>_VisionGuard.pdf`.
