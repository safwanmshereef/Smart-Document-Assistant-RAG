FROM python:3.11-slim

# Set environment variables to optimize Python container execution
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install basic compiler tools for pip compilation tasks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir beautifulsoup4

# Copy codebase
COPY . .

# Expose ports for both services
EXPOSE 8000
EXPOSE 8501
