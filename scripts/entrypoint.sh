#!/bin/bash
set -e

# 0. SSH 설정 강제 변경 (sshd_config 해킹)
if [ -f /etc/ssh/sshd_config ]; then
    echo "🔓 Unlocking SSH Root Login..."
    
    # 주석(#)이 있든 없든 'PermitRootLogin yes'로 무조건 치환
    sed -i 's/^#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
    sed -i 's/^PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
    
    # 키 인증 허용
    sed -i 's/^#PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config
    
    # (옵션) 비밀번호 인증도 일단 켜둠 (비상용)
    sed -i 's/^#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
else
    echo "⚠️ Error: /etc/ssh/sshd_config not found. Is openssh-server installed?"
fi

# 0. RunPod이 던져준 SSH 키를 등록하는 로직
if [ -n "$PUBLIC_KEY" ]; then
    echo "🔑 Setting up SSH access..."
    mkdir -p /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
    
    # SSH 서비스가 설치되어 있다면 재시작 (혹시 몰라서)
    if service ssh status > /dev/null 2>&1; then
        service ssh start
    fi
else
    echo "⚠️ Warning: No PUBLIC_KEY environment variable found."
fi

# 0. ssh 서비스 시작
echo "🔄 Starting SSH Service..."
if service ssh status > /dev/null 2>&1; then
    service ssh restart
else
    service ssh start
fi

echo "🚀 Starting Distributed Llama Container..."

# 1. Prepare Persistent Volume
# RunPod typically uses /workspace as the persistent volume mount point.
if [ -d "/workspace" ]; then
    echo "💾 Volume detected at /workspace"
    mkdir -p /workspace/models
    
    # If /app/models exists as a directory (from build), replace it with a symlink
    if [ -d "/app/models" ] && [ ! -L "/app/models" ]; then
        echo "🔗 Linking /app/models -> /workspace/models"
        rm -rf /app/models
        ln -s /workspace/models /app/models
    elif [ ! -e "/app/models" ]; then
        echo "🔗 Linking /app/models -> /workspace/models"
        ln -s /workspace/models /app/models
    fi
else
    echo "⚠️  No persistent volume found at /workspace. Using local /app/models (Data will be lost on restart!)"
    mkdir -p /app/models
fi

# 2. Check for Models and Auto-download
# If the models directory is empty, run the download script.
if [ -z "$(ls -A /app/models)" ]; then
    echo "📥 No models found. Downloading default models..."
    if [ -f "./prepare_models.sh" ]; then
        ./prepare_models.sh
    else
        echo "❌ prepare_models.sh not found!"
    fi
else
    echo "✅ Models detected in /app/models"
fi

# 3. Execute Command
# If arguments are provided (e.g., ./hpipe-worker ...), execute them.
# Otherwise, start a bash shell.
if [ "$#" -eq 0 ]; then
    exec /bin/bash
else
    exec "$@"
fi
