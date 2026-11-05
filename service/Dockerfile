# icdassist: the coder-facing service and the CLI that fits its model.
#
# Intentionally not an ML image. No CUDA base, no torch, no model weights -- the encoder runs in its
# own container (see docker-compose.yml) because a site may already operate one, may want a
# different one, and will certainly want to size its GPU separately from this process.
FROM python:3.12-slim

# No compiler, no git, nothing that is not needed to run numpy and open a socket.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY icdassist/ ./icdassist/
COPY web/ ./web/
COPY demo/ ./demo/

# Runs as a non-root user: this process reads a bundle and writes nothing but logs.
RUN useradd --create-home --uid 10001 icd && chown -R icd:icd /app
USER icd

# A bundle is mounted here; /data is where a site keeps its notes and embeddings.
VOLUME ["/bundle", "/data"]
EXPOSE 8080

# 0.0.0.0 inside the container is correct -- the compose file decides what the host exposes.
ENV ICDASSIST_EMBED_URL=http://ollama:11434/api/embed
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=2).status==200 else 1)"

ENTRYPOINT ["python", "-m", "icdassist"]
CMD ["serve", "--bundle", "/bundle", "--host", "0.0.0.0", "--port", "8080"]
