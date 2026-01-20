# Use Python 3.12 slim for compatibility with pyproject.toml
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing and webrtcvad
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY . .

# Install Python dependencies using uv sync (respects uv.sources for git deps)
RUN uv sync --frozen --no-dev

# Expose HuggingFace Spaces default port
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the Streamlit app using uv run
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
