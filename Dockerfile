# syntax=docker/dockerfile:1
FROM python:3.10-bullseye AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for scientific Python stacks
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY setup.py README.md ./
COPY bio_nav_data ./bio_nav_data
COPY main.py ./main.py

# Install package
RUN pip install .

# Create output directories to avoid permission issues at runtime
RUN mkdir -p /app/output /app/plots /app/logs

VOLUME ["/app/output", "/app/plots", "/app/logs"]

ENTRYPOINT ["python", "main.py"]
