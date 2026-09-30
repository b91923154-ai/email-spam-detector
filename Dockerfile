FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY . .

# Download NLTK data at build time
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)"

# Expose ports: 8000 for REST API, 8501 for Streamlit Web UI
EXPOSE 8000 8501

# Default execution: launch FastAPI REST service
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
