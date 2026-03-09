FROM python:3.12-slim

WORKDIR /app

# Copy project metadata and source for dependency installation.
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

# Copy remaining files (policy.yaml, etc.).
COPY . .

CMD ["python", "-m", "src.app.main"]
