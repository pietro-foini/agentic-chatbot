FROM python:3.14-slim AS builder

WORKDIR /app

# Install Poetry.
RUN pip install --no-cache-dir poetry==2.5.1

# Copy dependency files.
COPY pyproject.toml poetry.lock* ./

# Configure poetry to create the virtual environment in the project root (.venv).
RUN poetry config virtualenvs.in-project true && poetry install --only main --no-root

FROM python:3.14-slim AS runner

WORKDIR /app

# Copy built virtual environment from the builder stage.
COPY --from=builder /app/.venv /app/.venv

# Copy application source code.
COPY chatbot/ ./chatbot/

# Copy scripts.
COPY scripts/ ./scripts/

# Add the virtual environment to PATH.
ENV PATH="/app/.venv/bin:$PATH"

# Expose FastAPI backend port.
EXPOSE 8000

# Run Uvicorn directly as the primary process.
CMD ["sh", "-c", "python -m scripts.setup && uvicorn chatbot.main:app --host 0.0.0.0 --port 8000"]