# Use Python 3.11 slim image for smaller size and security
FROM python:3.11-slim

# Set environment variables for Python optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd -r botuser && useradd -r -g botuser botuser

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
# Install dependencies with preference for binary wheels
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy application code
COPY agentic_gemini_bot.py .

# Set proper permissions for non-root user (no responses directory needed - memory storage)
RUN chown -R botuser:botuser /app

# Switch to non-root user
USER botuser

# Expose port for Flask API
EXPOSE 5001

# Environment variables for secrets (to be provided at runtime via secrets or .env)
ENV BOT_TOKEN="" \
    GEMINI_API_KEY="" \
    TELEGRAM_BOT_TOKEN=""

# Environment variables for bot configuration (with sensible defaults)
ENV MAX_CONCURRENT_USERS=100 \
    MAX_CONVERSATIONS_PER_USER=20 \
    USER_SESSION_TIMEOUT_MINUTES=60 \
    MAX_MEMORY_USAGE_MB=500 \
    FLASK_HOST="0.0.0.0" \
    FLASK_PORT="5001"

# Health check using curl (more reliable than Python requests)
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5001/api/health || exit 1

# Run the bot
CMD ["python", "agentic_gemini_bot.py"] 