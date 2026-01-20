# ==============================================================================
# Pharma Voice Orders - Hugging Face Spaces Dockerfile
# ==============================================================================
# This Dockerfile is optimized for Hugging Face Spaces with Docker SDK.
# It uses standard pip for maximum compatibility.
# ==============================================================================

FROM python:3.11-slim

# Set working directory FIRST
WORKDIR /app

# Install system dependencies for audio processing and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create non-root user for security (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory for user
WORKDIR $HOME/app

# Copy files to user directory
COPY --chown=user . $HOME/app

# Expose HuggingFace Spaces default port
EXPOSE 7860

# Set environment variables for Streamlit
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
