FROM python:3.14-slim AS builder

WORKDIR /app

# Copy dependency files.
COPY requirements.txt pyproject.toml poetry.lock* ./

# Configure poetry to create the virtual environment in the project root (.venv).
RUN pip install --no-cache-dir -r requirements.txt
RUN poetry config virtualenvs.in-project true \
    && poetry install --only main --no-root

FROM python:3.14-slim AS runner

WORKDIR /app

# Copy built virtual environment from the builder stage.
COPY --from=builder /app/.venv /app/.venv

# Copy application source code.
COPY chatbot/ ./chatbot/

# Add the virtual environment to PATH.
ENV PATH="/app/.venv/bin:$PATH"

# Expose FastAPI backend port.
EXPOSE 8000

# Run Uvicorn directly as the primary process.
CMD ["uvicorn", "chatbot.main:app", "--host", "0.0.0.0", "--port", "8000"]