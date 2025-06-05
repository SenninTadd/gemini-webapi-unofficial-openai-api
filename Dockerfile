# Use Python 3.8+ as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api_server.py .
COPY run_server.py .
COPY gemini_openai.py .
COPY README.md .

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables for uvicorn
ENV HOST=0.0.0.0
ENV PORT=8000

# Command to run the application using uvicorn directly
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"] 