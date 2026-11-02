# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed for some Python packages (e.g. chromadb native libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first (layer-cache friendly)
COPY pyproject.toml ./

# Install into a local prefix so we can copy it to the runtime image
RUN pip install --upgrade pip \
 && pip install --prefix=/install . \
 && rm -rf /root/.cache/pip


# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# curl needed by Docker health check
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source
COPY src/    ./src/
COPY main.py ./

# Create a non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

EXPOSE 8080

CMD ["python", "main.py"]
