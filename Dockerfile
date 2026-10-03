# TensorForge ticket classifier API.
#
#   docker build -t tensorforge .
#   docker run -p 8000:8000 -e API_KEY=<key> tensorforge
#
# Needs no internet at runtime: the model is copied into the image.
# Python 3.13 matches the version the model was trained with (joblib pickles are version-sensitive).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

# Run as a normal user; job database lives in a writable folder.
RUN useradd --create-home --uid 1000 app && mkdir -p /srv/state && chown app /srv/state
ENV JOBS_DB=/srv/state/jobs.db

COPY app/ app/
COPY model/ model/

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# One worker process: async job state is shared through SQLite and processed by one background thread.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
