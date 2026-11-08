# VisionGuard web app / model API image.
# - slim base + CPU-only TensorFlow (see requirements-app.txt) keeps the image small
# - dependencies are copied+installed BEFORE the source so the (slow) pip layer is cached and only
#   rebuilt when requirements-app.txt changes, not on every code edit
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 FLASK_DEBUG=0 MODEL_PATH=/models/champion/model.keras

RUN useradd --create-home --uid 10001 appuser
WORKDIR /srv

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

COPY visionguard ./visionguard
COPY app ./app
COPY api ./api

USER appuser
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','5000'), timeout=4)"

# No secrets here: DATABASE_URL, API keys, ... are injected at run time (compose .env).
CMD ["flask", "--app", "app.app:create_app", "run", "--host", "0.0.0.0", "--port", "5000"]
