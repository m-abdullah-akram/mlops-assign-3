# Tasks 7 and 8 - deploying the model API and integrating it with Flask

**Architecture.** The cloud VM runs only the model API (`api/api_server.py`, TensorFlow + the champion
model). Your Flask UI runs on `localhost`, forwards each uploaded image to the API with an `X-API-Key`
header, and writes the result into its own database (the history page). This matches the brief
("model on the cloud, Flask on localhost") and is Pathway A/C (a VM running Docker).

## A. Provision the VM (Oracle Always Free shown; AWS/GCP/Azure are the same from step 3)
1. Create an Ubuntu 22.04 instance. **Research and write down in the report:** the free limits shown in your
   console, Arm (Ampere A1, `aarch64`) vs x86 (AMD micro), instance availability, idle-reclamation rules.
   TensorFlow needs roughly 1.5-2 GB RAM: the 1 GB AMD micro is too small, so use an Ampere A1 shape
   (2+ GB). On Arm confirm `pip install tensorflow-cpu==2.15.1` resolves inside the build; if it has no
   aarch64 wheel, change `requirements-app.txt` to `tensorflow==2.15.1` for that VM and note it in the report.
2. Open **both firewalls** for TCP 8000 (and 22 for SSH):
   - Cloud network rule: VCN > Security List (or NSG) > Ingress rule, source `0.0.0.0/0`, port `8000`.
   - The VM's own firewall: `sudo iptables -I INPUT 6 -p tcp --dport 8000 -j ACCEPT && sudo netfilter-persistent save`
     (Ubuntu on Oracle ships restrictive iptables rules; with ufw use `sudo ufw allow 8000/tcp`).
3. Install Docker: `curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER` (log out/in).

## B. Ship code and model
```bash
git clone https://github.com/mabdullah-akram/22l-6632_A2_Ops.git && cd 22l-6632_A2_Ops
# the model is not in Git: copy models/champion from your machine (created by export_champion)
scp -r models/champion ubuntu@YOUR_SERVER_IP:~/22l-6632_A2_Ops/models/
printf "API_KEY=%s\nCOMPOSE_PROJECT_NAME=22l_6632_a2_ops\n" "$(openssl rand -hex 24)" > .env
docker compose -f docker-compose.api.yml up -d --build
docker compose -f docker-compose.api.yml logs -f api      # look for "Loaded model ... (threshold=...)"
curl http://localhost:8000/health
```
Keep the generated `API_KEY` (`grep API_KEY .env`); it is the shared secret.

## C. Test from Postman (from a different network, e.g. phone hotspot)
- `GET http://YOUR_SERVER_IP:8000/health` -> `{"status":"ok","model_loaded":true,...}`
- `POST http://YOUR_SERVER_IP:8000/predict`, Headers: `X-API-Key: <key>`, Body > form-data > key `file`
  (type **File**) > choose a JPG/PNG -> JSON with `label`, `confidence`, `prob_ai`, `model_version`.
- Also capture: no key -> `401`, wrong file -> `400`. Screenshot request + response + the
  `docker compose logs api` line (`PREDICT file=... label=...`) as the "logged result".

## D. Integrate with the local Flask app (Task 8)
```bash
export INFERENCE_MODE=remote REMOTE_API_URL=http://YOUR_SERVER_IP:8000 REMOTE_API_KEY=<key>
flask --app "app.app:create_app" run --port 5000
```
(Or put the three values in `.env` and use `docker compose up` locally.) Upload several different images
at http://localhost:5000, screenshot each result, then open `/history`: the **Source** column reads
`remote-api` and shows model version, decision, confidence and time.

## Security notes (mention in the report)
- Plain HTTP sends the API key unencrypted. For real use put Caddy/nginx with TLS in front (needs a domain)
  or use a tunnel. Using HTTP here is an accepted assignment trade-off; say so.
- The key lives only in `.env` on the server and the client; `.env` is git-ignored.
- Rate limiting is not implemented (out of scope); the 5 MB limit and file validation bound abuse.

## Pathway B (Vercel) - why this repo does not use it
TensorFlow plus a model exceeds Vercel's serverless function size limit, functions have no persistent
disk (SQLite would not persist) and have short timeouts, so Vercel could only host a thin UI that calls a
model server elsewhere - which is exactly what the Flask-localhost + VM API design already does.
