# VisionGuard - MLOps Assignment 2

**Name:** ______  **Roll number:** ______
**GitHub:** https://github.com/mabdullah-akram/22l-6632_A2_Ops  **DagsHub:** https://dagshub.com/ma92365/22l-6632_A2_Ops
**Live API URL:** http://YOUR_SERVER_IP:8000  **Graded commit:** `git rev-parse HEAD` -> ______
**AI-assistant disclosure:** I used an AI assistant (Claude) to generate the initial project code,
configuration and a report outline. I ran every command myself, reviewed and understood the code, and wrote
the observations and screenshots from my own runs. (Edit this to be accurate for you.)

> How to use: for each sub-task keep the three headings *What I did / Why / What went wrong*. The text in
> **[draft]** blocks is a technically correct starting point - rewrite it in your own words and keep only
> what you actually did. Insert your screenshots under each heading with a caption ("Figure N: ...").

---
## Chapter 1 - Data understanding and versioning
**1.1 Environment.** What/why: isolated venv so package versions are reproducible; versions recorded for TensorFlow,
etc.; GPU check result. [Screenshot]
**1.2 Dataset audit.** Describe `visionguard/audit.py`: walks the folders, derives class from the folder name, records size,
width, height, mode, format, SHA-256; corrupt detection = `Image.verify()` + full `load()`; duplicates = same SHA-256.
Insert counts table and min/median/max table; state corrupt and duplicate counts. Explain that duplicates are removed
before splitting because the same image in train and test would inflate test scores (data leakage).
**1.3 Visual EDA.** Insert both figures, each with 2-3 sentences (e.g. do image sizes or file sizes differ systematically
between classes? Such differences are shortcuts a model could learn instead of real AI artefacts).
**1.4 DVC.** Toy part: steps and screenshots. **[draft]** The pointer file (`.dvc`) is a tiny YAML text file holding
the content hash (md5), size and file count of the tracked data and its path. Git versions this pointer; DVC stores the
actual bytes in its cache and remote. Large binaries do not belong in Git because every version is kept forever in the
history, bloating the repository, making clones slow, and hitting hosting size limits; Git also cannot diff or delta
binary data well. Apply part: dataset tracked, remote = DagsHub storage, `dvc push` screenshot.

## Chapter 2 - Model development
**2.1 Input pipeline.** **[draft]** Stages: read split CSV -> read file bytes -> `tf.io.decode_image` (3 channels) ->
resize to 224x224 -> (train only) shuffle -> batch(32) -> random flip / rotation / zoom / contrast -> `prefetch(AUTOTUNE)`.
224 is MobileNetV2's native size and cheap on CPU. Resize stretches the image (aspect ratio not preserved) - this is
simple and consistent between training and serving; the cost is mild distortion. Augmentation only on train so
validation/test measure the real data distribution. Insert `augmented_batch.png`.
**2.2 Baseline CNN.** Table of layers (output shape, params) from `model_summary.txt`; justify each layer: Rescaling
(0-1 inputs), Conv (learn local filters), BatchNorm (stabilise training), ReLU (non-linearity), MaxPool (reduce size /
add translation tolerance), GlobalAveragePooling (few parameters vs Flatten), Dropout (regularise), Dense(1, sigmoid) (probability
of AI). Loss = binary cross-entropy (probabilistic binary target); Adam (robust default, adaptive steps);
metrics accuracy, AUC, precision, recall.
**2.3 Transfer learning.** MobileNetV2, `include_top=False`, ImageNet weights. Stage 1: base frozen, only head
(GAP + Dropout + Dense) trains. Stage 2: only the last block (`block_16_*`) unfrozen, BatchNorm layers kept frozen, lr
1e-5 (much smaller than 1e-3) to avoid destroying pretrained features. Preprocessing: MobileNetV2 was trained on inputs
scaled to [-1, 1], implemented as a Rescaling layer inside the model. Include both curve sets and the two
trainable-layer lists.
**2.4 Training discipline.** Seed fixed with `tf.keras.utils.set_random_seed(42)`; EarlyStopping(val_loss, patience 4,
restore best), ModelCheckpoint(val_loss), ReduceLROnPlateau(factor 0.5). State best epoch per stage.
**2.5 Evaluation.** Metric table for both models, confusion matrices, ROC. Threshold chosen on validation by maximising
F-beta (beta = ___ , the value you passed to `--beta`). **Which error is worse? (your decision)** Arguments: calling a
real photo fake wrongly accuses a genuine creator; calling a fake real lets misleading content through. State your
choice, justify it for this product, and make sure the beta you used matches it.
**2.6 Error analysis.** Insert `misclassified.png`; one hypothesis per image (blur, painting style, watermark, size,
compression, ...).
**2.7 Keras Tuner.** Toy table, VisionGuard table (>= 8 trials). **[draft]** Which setting mattered most: read it
from your table. Picking the best validation score can overfit the validation set because with many trials the
winner is partly the one that got lucky on that particular sample (selection bias); the validation score of the winner is
optimistically biased, which is why the untouched test set is used once at the end.
**Bonus.** Grad-CAM figure and what the model looks at (object, texture, background, edges?).

## Chapter 3 - MLflow and DagsHub
**3.1** Definitions in your words: *experiment* (a named group of related runs), *run* (one execution of training
with its own logs), *parameter* (an input setting fixed for the run, e.g. lr), *metric* (a numeric result, optionally per
step, e.g. val_loss), *artifact* (a file produced by the run: plot, model), *tag* (free-form label for searching, e.g.
git commit). Screenshots of two runs + comparison.
**3.2** DagsHub project, tracking URI, token via environment variables (`.env`), proof: `git check-ignore -v .env`.
**3.3** Manual (`MlflowEpochCallback` + explicit params/tags/test metrics) vs `mlflow.tensorflow.autolog`. **[draft]**
Manual captures exactly what you choose, including both stages on one step axis, tags (git commit, DVC data version,
seed) and test metrics. Autolog captures Keras params, optimizer settings and per-epoch metrics with no code, but it
does not know your data version, git commit or stage structure (epochs restart in stage 2), and it does not log
test-set metrics because evaluation happens outside `fit`; those still need explicit calls.
**3.4** Artifacts screenshot. **3.5** Comparison table of >= 6 runs, each changing one factor; choose the best and
justify (e.g. ROC-AUC and recall of the class you decided matters most, plus val-test gap). **3.6** Registry versions,
alias `champion`, `predict_champion` output.

## Chapter 4 - Flask application
**4.1** **[draft]** The model is loaded once at start-up because loading reads the file and rebuilds the graph (seconds)
and duplicates memory per request. Preprocessing must match training exactly (same decode, same resize, same scaling)
or the model receives a different input distribution and accuracy silently drops; here one shared function
(`visionguard/preprocess.py`) and the Rescaling layer inside the model enforce this. Console output for two images.
**4.2** Validation order: extension whitelist -> `secure_filename` -> size limit (413) -> decode check with Pillow
(catches renamed/corrupt files) -> nothing saved to disk (processed in memory). Three failure screenshots.
**4.3** Upload, result and history pages. **4.4** Settings come from environment variables; debug off. **[draft]** Debug
mode exposes the Werkzeug interactive debugger (remote code execution) and the reloader starts the app twice
(loading the model twice).

## Chapter 5 - Database trail
SQLite locally, PostgreSQL in Compose; table `predictions(id, created_at, filename, label, confidence, prob_ai,
threshold, model_name, model_version, source, latency_ms)`; a row is written after every successful inference.
Screenshots: history page and a `psql` query.

## Chapter 6 - Docker
**6.1** Definitions: *image* (read-only template with the filesystem and start command), *container* (running instance
of an image), *layer* (cached filesystem diff per Dockerfile instruction), *volume* (storage that outlives the container),
*port mapping* (`-p host:container`). Screenshots of each action.
**6.2** **[draft]** `requirements-app.txt` is copied and installed before the source: Docker caches layers and reuses a
layer only if everything above it is unchanged, so editing code does not re-download TensorFlow. Slim base, `tensorflow-cpu`
(no GPU libraries; the server has no GPU), non-root `appuser`, `HEALTHCHECK`, `.dockerignore`, no secrets in the image.
**6.3** Compose: `db` (named volume `pgdata`, health check, no published port) and `web` (`depends_on: service_healthy`,
reaches the DB at host `db`). Model is **mounted** read-only from `./models`: keeps the image small, lets you swap the champion without
rebuilding, and avoids baking a large artifact into every image. Show `docker compose ps` healthy and the closed DB port.
**6.4** `docker images` sizes before (`Dockerfile.unoptimized`) and after, `docker history` largest layers, conclusion.
**6.5** Operations: `up -d`, `stop`, `start`, `restart`, `logs`, `exec web sh`, `exec db sh`. Break: wrong `POSTGRES_PASSWORD` in
`.env` after the volume exists, or remove `models/champion`; show the log lines, the diagnosis, the fix.
**6.6** Toy red/green runs, project workflow (`.github/workflows/ci.yml`), badge. **[draft]** Secrets are stored in
repository Settings > Secrets and variables > Actions (encrypted, masked in logs); a workflow file is committed and
readable by everyone, so a secret written there would leak through the repository history.

## Chapter 7 - API deployment
Step-by-step with screenshots following `docs/DEPLOYMENT.md`; API URL; Postman GET/POST results; API log line;
current free-tier limits you researched.

## Chapter 8 - Flask + API integration
Several uploads through the local Flask app calling the cloud API; screenshots of results and of the history page
showing `remote-api` entries.

## Submission checklist
- [ ] `<RollNumber>_VisionGuard.pdf` (page limit respected)  - [ ] DagsHub link public/shared
- [ ] API URL tested from another network  - [ ] README badge visible  - [ ] `.env` not in Git
