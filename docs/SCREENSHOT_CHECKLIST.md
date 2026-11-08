# Screenshot and evidence checklist

Every screenshot must show your username/prompt, date/time and project name, be captioned and referenced
in the text. "OK" = tick when captured. Nothing here is pre-made: produce each from your own runs.

| ID | Evidence to capture | OK |
|----|---------------------|----|
| 1.1 | Terminal: Python + TF versions, `pip freeze` excerpt, GPU list (empty is fine - say so) | |
| 1.2 | `audit_summary.txt` output: counts per class, min/median/max, corrupt, duplicates | |
| 1.3 | `sample_grid.png`, `histograms.png` + 2-3 sentence caption each | |
| 1.4 | Toy: `git log`, `dvc add`, v1/v2, checkout old, push, delete, pull. Apply: `dvc add data/raw`, DagsHub files view, commits | |
| 2.1 | `augmented_batch.png` + pipeline description | |
| 2.2 | `outputs/<run>/model_summary.txt` screenshot + your own per-layer table | |
| 2.3 | Two curve sets (`curves.png` has both stages), `trainable_layers_stage1/2.txt` | |
| 2.4 | `curves.png` with best epoch marked, best epoch printed in terminal | |
| 2.5 | Metric table (baseline vs transfer from `metrics.json`), confusion matrix, ROC | |
| 2.6 | `misclassified.png` + one hypothesis per image | |
| 2.7 | Toy tuner results table; `reports/tuner_results.json` table + discussion | |
| Bonus | `reports/figures/gradcam.png` + comment | |
| 3.1 | Local MLflow UI: both toy runs + comparison view | |
| 3.2 | DagsHub project page, toy run on DagsHub, `git check-ignore -v .env` output | |
| 3.3 | Run page for `mnv2-ref` (manual) and `mnv2-autolog`: params, metric curves, tags | |
| 3.4 | Artifacts tab (confusion matrix, ROC, splits, requirements_frozen.txt, model) | |
| 3.5 | Experiment comparison table/chart with >= 6 runs | |
| 3.6 | Registry page (2 versions, alias `champion`), terminal output of `predict_champion` | |
| 4.1 | Console output for two known images (one real, one AI) | |
| 4.2 | Screenshots: wrong file type, too large, corrupt "fake" jpg | |
| 4.3 | Upload page, result page, history page | |
| 4.4 | `curl /health` output, debug-off explanation | |
| 5 | History page after several requests; `docker compose exec db psql ... "select * from predictions"` | |
| 6.1 | One screenshot per Docker action + definitions | |
| 6.2 | `docker compose exec web whoami` (or `id`) -> `appuser`, Dockerfile link | |
| 6.3 | `docker compose ps` showing healthy; from the host `nc -zv localhost 5432` refused/failing | |
| 6.4 | `docker images` + `docker history` before (Dockerfile.unoptimized) and after | |
| 6.5 | start/stop/restart/logs/shell for both services; broken-password log excerpt; fix | |
| 6.6 | Toy red + green runs; project green run; badge in README | |
| 7 | VM console, firewall rules, Docker running, Postman GET/POST, API URL, API logs | |
| 8 | 3+ different images through local Flask -> remote API; history page with `remote-api` rows | |
| Front matter | name, roll number, links, `git rev-parse HEAD` of graded commit, AI-use disclosure | |
