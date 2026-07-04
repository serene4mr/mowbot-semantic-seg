FROM ultralytics/ultralytics:latest

# Install system dependencies needed for VS Code Remote Development and basic workflow.
# VS Code requires 'procps' (which provides 'ps') to monitor and manage processes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Keep Python output unbuffered to see logs in real-time
ENV PYTHONUNBUFFERED=1
