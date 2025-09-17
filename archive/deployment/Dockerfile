FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Configure GitHub access for private repositories
RUN --mount=type=secret,id=github_token \
    export GITHUB_TOKEN=$(cat /run/secrets/github_token) && \
    git config --global url."https://$GITHUB_TOKEN@github.com/".insteadOf "https://github.com/"

# Copy requirements
COPY pyproject.toml ./

# Install Poetry
RUN pip install poetry

# Configure Poetry
RUN poetry config virtualenvs.create false

# Install dependencies with GitHub access
RUN --mount=type=secret,id=github_token \
    export GITHUB_TOKEN=$(cat /run/secrets/github_token) && \
    poetry config repositories.github-spi "https://$GITHUB_TOKEN@github.com/OmniNode-ai/omnibase_spi.git" && \
    poetry config repositories.github-core "https://$GITHUB_TOKEN@github.com/OmniNode-ai/omnibase_core.git" && \
    poetry config repositories.github-infra "https://$GITHUB_TOKEN@github.com/OmniNode-ai/omnibase_infra.git" && \
    pip install "git+https://$GITHUB_TOKEN@github.com/OmniNode-ai/omnibase_spi.git@main" && \
    pip install "git+https://$GITHUB_TOKEN@github.com/OmniNode-ai/omnibase_core.git@main" && \
    pip install "git+https://$GITHUB_TOKEN@github.com/OmniNode-ai/omnibase_infra.git@main" && \
    poetry install --only=main --no-root

# Copy source code
COPY src/ /app/src/

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Create non-root user for security
RUN useradd -m -u 1000 omnimemory && chown -R omnimemory:omnimemory /app
USER omnimemory

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "omnimemory.api.main:app", "--host", "0.0.0.0", "--port", "8000"]