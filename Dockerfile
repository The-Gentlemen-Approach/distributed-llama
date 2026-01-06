# 1. Base Image: NVIDIA CUDA Devel (Ubuntu 22.04)
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# 2. Set Environment Variables
ENV DEBIAN_FRONTEND=noninteractive

# 3. Install Basic Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python3 \
    python3-pip \
    wget \
    gnupg \
    curl \
    ca-certificates \
    libvulkan1 \
    vulkan-tools \
    libvulkan-dev \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Vulkan SDK (for glslc)
RUN wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | gpg --dearmor -o /usr/share/keyrings/lunarg-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/lunarg-archive-keyring.gpg] https://packages.lunarg.com/vulkan/ jammy main" | tee /etc/apt/sources.list.d/lunarg-vulkan-jammy.list && \
    apt-get update && \
    apt-get install -y vulkan-sdk && \
    rm -rf /var/lib/apt/lists/*

# 5. Create Work Directory
WORKDIR /app

# 6. Setup Entrypoint Script
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 7. Copy Source Code & Build
COPY . /app
RUN make clean && \
    make hpipe-root hpipe-worker DLLAMA_VULKAN=1

# 9. Expose Ports
EXPOSE 9999

# 10. Define Entrypoint & Default Command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["sleep", "infinity"] 
# CMD ["/bin/bash", "-c", "sleep infinity"]