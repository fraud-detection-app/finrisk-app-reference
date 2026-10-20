# The risk service, packed as a container image.
#
#   docker build -t finrisk-service:dev .
#   docker run -p 8000:8000 finrisk-service:dev      # /ready says 503 here: no model is baked into THIS image
#
# This is the SERVICE image: code and libraries. The release image (Dockerfile.aws) is built on top of it
# and adds the small baked fallback model and index. The ML repo's training image is built on top of the
# release image too, so training and serving use the very same feature and model code.
#
# Read it top to bottom: each instruction adds one LAYER to the image, and Docker caches layers
# in order. The things that change least come first, so editing your code does not re-install
# 1.3 GB of libraries.

# 1. Start from a small Linux with Python already installed. Pin the version: "latest" changes
#    under you. 3.12 is what AWS's own Python images use for new work.
FROM python:3.12-slim

# 2. Settings for every process in the container.
#    PYTHONUNBUFFERED: print logs immediately (a crashed container cannot flush a buffer).
#    PYTHONPATH: lets `import financial_risk` find our package.
#    PORT: where the server listens. Cloud platforms often choose it, so read it from here.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PORT=8000

WORKDIR /app

# 3. Dependencies first. This slow layer (~1.3 GB) is rebuilt ONLY when this file changes.
COPY requirements-service.txt ./
RUN pip install --no-cache-dir -r requirements-service.txt

# 4. Then our code, which changes on every commit. Rebuilding from here takes about a second.
COPY VERSION ./VERSION
COPY src ./src
COPY api ./api
COPY scripts ./scripts
COPY config ./config
COPY rag_docs ./rag_docs
COPY docs ./docs

# What is NOT copied: data/, models/, vector_index/ and .env (see .dockerignore).
# In AWS the model and the knowledge index are NOT in the image at all: the service reads two SSM
# pointers (POINTER_PREFIX) and downloads what they name, so a new model or index reaches an
# environment without a new image. Locally you can mount a folder instead:
#   docker run -p 8000:8000 -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" \
#              -v "$PWD/vector_index:/app/vector_index" finrisk-service:dev

# 5. Documentation, not a firewall rule: EXPOSE records the port; `-p` actually publishes it.
EXPOSE 8000

# 6. Docker's own health check: `docker ps` shows (healthy) once /health answers. AWS services use
#    their own checks (configured on the service), but this one is free while developing.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'], timeout=2)"

# 7. The command the container runs. 0.0.0.0, NOT 127.0.0.1: inside a container, 127.0.0.1 means
#    "this container only", so a published port would reach nothing -- the most common first-
#    container mistake. `exec` makes uvicorn process 1, so `docker stop` shuts it down cleanly.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
