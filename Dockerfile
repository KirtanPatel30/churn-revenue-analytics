FROM python:3.11-slim

WORKDIR /app

# System deps needed for xgboost/shap wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the pipeline once at build time so the dashboard has data/model
# artifacts baked into the image (avoids re-training on every restart).
RUN python run_pipeline.py

EXPOSE 8050

# Shell form so $PORT (set by Render at runtime) is actually expanded.
# Falls back to 8050 for local `docker run` testing.
CMD gunicorn --chdir src -b 0.0.0.0:${PORT:-8050} app:server
