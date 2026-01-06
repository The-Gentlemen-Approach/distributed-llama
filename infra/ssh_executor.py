import os
import paramiko
import time
from typing import Optional, List

class SSHExecutor:
    """Responsibility 2: Handle SSH command execution independently of the resource provider."""
    def __init__(self, host: str, port: int, username: str = "root", key_path: Optional[str] = None):
        self.host = host
        self.port = port
        self.username = username
        self.key_path = os.path.expanduser(key_path or "~/.ssh/id_rsa")
        self.client: Optional[paramiko.SSHClient] = None

    def connect(self, retries: int = 10, delay: int = 5):
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        print(f"📡 Connecting to {self.host}:{self.port} (Max {retries} attempts)...")
        for i in range(retries):
            try:
                self.client.connect(
                    self.host, 
                    port=self.port, 
                    username=self.username, 
                    key_filename=self.key_path,
                    timeout=10
                )
                print(f"✅ SSH Connection established to {self.host}:{self.port}")
                return
            except Exception as e:
                if i == retries - 1:
                    print(f"❌ Failed to connect after {retries} attempts.")
                    raise e
                print(f"   [Attempt {i+1}] Connection not ready, retrying in {delay}s...")
                time.sleep(delay)

    def run(self, command: str, background: bool = False) -> List[str]:
        if not self.client:
            self.connect()
        
        if background:
            # For background, we assume the command already has its own redirection
            self.client.exec_command(command)
            return []
        else:
            stdin, stdout, stderr = self.client.exec_command(command)
            out_lines = stdout.readlines()
            err_lines = stderr.readlines()
            
            if err_lines:
                print(f"⚠️ [SSH STDERR] {self.host}: {''.join(err_lines).strip()}")
            return out_lines

    def stream_logs(self, command: str, prefix: str = ""):
        if not self.client:
            self.connect()
            
        stdin, stdout, stderr = self.client.exec_command(command)
        for line in iter(stdout.readline, ""):
            print(f"[{prefix}] {line.strip()}")

    def download_file(self, remote_path: str, local_path: str):
        """Downloads a file from the remote pod via SFTP."""
        if not self.client:
            self.connect()
            
        sftp = self.client.open_sftp()
        try:
            print(f"📥 Downloading {remote_path} to {local_path}...")
            sftp.get(remote_path, local_path)
        finally:
            sftp.close()

    def close(self):
        if self.client:
            self.client.close()
