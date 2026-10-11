# mivia-small-json serving image: REST quartet + MCP over streamable HTTP.
#
# Deploys the shipped model over the internet on CPU (< 2 GB RAM). The
# build context is intentionally narrow: only src/, the model artifacts,
# and the pin files enter the image — never data/ (the gitignored
# owner-repo harvest under data/incoming/gh-user/ can therefore never
# leak into a published image), never .git, never training extras (no
# torch; runtime stays CPU-only).
#
# Build:  docker build -t mivia-small-json .
# Run:    docker run -d -p 8000:8000 -e MIVIA_API_TOKEN=change-me \
#             --name mivia mivia-small-json
# MCP:    claude mcp add --transport http mivia http://HOST:8000/mcp \
#             --header "Authorization: Bearer change-me"
# Health: curl http://localhost:8000/health   -> {"status": "ok", ...}
#
# Behind a reverse proxy (TLS termination lives there, not here), set
# MIVIA_ALLOWED_HOSTS=api.example.com so the MCP transport's DNS-rebinding
# guard accepts the public Host header.

FROM python:3.12-slim

# OS trust store stays installed: the MCP SDK's HTTP client verifies TLS
# through the system store (truststore), and ca-certificates must survive
# in slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user; the models/ dir is chowned read-only to it.
RUN useradd --system --create-home --shell /usr/sbin/nologin mivia

WORKDIR /app

# Dependency layer first (cached across src changes): exact pins only,
# including the [serve] extra pins. requirements-lock.txt is copied for
# provenance/reproducibility of the runtime set.
COPY requirements.txt requirements-lock.txt pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir -r requirements.txt

# Source + artifacts. README.md/LICENSE are needed by the package metadata.
COPY src/ ./src/
COPY scripts/assemble_models.py ./scripts/
COPY models/ ./models/
# Split-volume artifacts (>100MB files ship as parts on GitHub): assemble
# and sha-verify before the app loads anything.
RUN python scripts/assemble_models.py
RUN pip install --no-cache-dir --no-deps -e . \
    && chown -R mivia:mivia /app \
    && chmod -R a-w /app/models

USER mivia

ENV MIVIA_HOST=0.0.0.0 \
    MIVIA_PORT=8000 \
    # Do NOT set this in production images: it loads gitignored local
    # builds from data/incoming/models/, which the image does not contain.
    MIVIA_ALLOW_STAGED=

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4); sys.exit(0 if r.status==200 else 1)"]

CMD ["python", "-m", "mivia_small_json.api", "--host", "0.0.0.0", "--port", "8000"]
