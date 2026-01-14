import os
import subprocess
import time
from typing import List, Optional
from runpod_pod import RunPodPod
from ssh_executor import SSHExecutor

class DistributedLlamaOrchestrator:
    """Responsibility 3: High-level orchestration for distributed-llama cluster."""
    def __init__(self, ssh_key: Optional[str] = None):
        self.worker_pods: List[RunPodPod] = []
        self.ssh_key = ssh_key
        self.executors: List[SSHExecutor] = []
        self.worker_configs = []  # 재사용 여부 추적용

    def setup_cluster(self, config: dict):
        print("🌐 Provisioning worker nodes on RunPod...")

        for i, w_cfg in enumerate(config['workers']):
            # 기존 pod_id가 있으면 재사용
            if 'pod_id' in w_cfg and w_cfg['pod_id']:
                print(f"♻️  Reusing existing worker node {i+1}: {w_cfg['pod_id']}")
                pod = RunPodPod(w_cfg['pod_id'])
            else:
                print(f"👷 Creating worker node {i+1}: {w_cfg['name']} ({w_cfg['gpu_type_id']})")
                pod = RunPodPod.create(
                    w_cfg['name'],
                    w_cfg['template_id'],
                    w_cfg['gpu_type_id']
                )
                print(f"   ✅ Created: {pod.pod_id}")
            self.worker_pods.append(pod)
            self.worker_configs.append(w_cfg)
        
        # Wait for all workers to be ready
        for pod in self.worker_pods:
            print(f"⏳ Waiting for {pod.pod_id}...")
            pod.wait_for_status("running")
        
        print("✅ All worker nodes are running.")
        print("🕯️ Giving SSH services 15 seconds to warm up...")
        time.sleep(15)

    def _get_ssh_port(self, pod: RunPodPod) -> int:
        return self._get_app_port(pod, 22)

    def _get_app_port(self, pod: RunPodPod, target_port: int = 9999) -> int:
        for p in pod.network_info['ports']:
            if p['containerPort'] == target_port:
                return p['publicPort']
        return target_port

    def run_inference(self, config: dict):
        settings = config.get('cluster_settings', {})
        model = settings.get('model_path')
        tokenizer = settings.get('tokenizer_path')
        threads = settings.get('nthreads', 16)
        steps = settings.get('steps', 50)
        prompt = settings.get('prompt', "Once upon a time")
        
        worker_endpoints = []
        self.executors = []
        
        # 1. Start Workers on RunPod (in parallel for faster startup)
        print("👷 Launching hpipe-workers on RunPod...")
        worker_configs = config.get('workers', [])

        # Phase 1: Build and prepare all workers in parallel
        print("📦 Phase 1: Building and preparing all workers...")
        worker_info = []

        for i, pod in enumerate(self.worker_pods):
            net = pod.network_info
            ssh_port = self._get_ssh_port(pod)
            app_port = self._get_app_port(pod, 9999)

            # Get worker specific config
            w_cfg = worker_configs[i] if i < len(worker_configs) else {}
            gpu_idx = w_cfg.get('gpu_index', 0)

            executor = SSHExecutor(net['public_ip'], ssh_port, key_path=self.ssh_key)
            self.executors.append(executor)

            # 1. Ensure binary exists
            base_dir = "/app"
            executor.run(f"mkdir -p {base_dir}")

            # Check for binary, build if missing or force rebuild with Vulkan
            # We always try to build with Vulkan=1 on workers since they are GPU pods
            print(f"   🔨 Building hpipe-worker on worker {i+1} with Vulkan...")
            build_result = executor.run(f"cd {base_dir} && make clean && make hpipe-worker DLLAMA_VULKAN=1 2>&1 | tee build.log")

            # Verify build success
            binary_check = executor.run(f"[ -f {base_dir}/hpipe-worker ] && echo 'exists' || echo 'missing'")
            if not binary_check or 'missing' in binary_check[0]:
                print(f"   ❌ Build failed on worker {i+1}!")
                print(f"   Last 20 lines of build output:")
                build_log = executor.run(f"tail -20 {base_dir}/build.log")
                if build_log:
                    for line in build_log:
                        print(f"      {line}")
                raise RuntimeError(f"Failed to build hpipe-worker on worker {i+1}")
            print(f"   ✅ Build successful on worker {i+1}")

            # 2. Check for model
            check_model_cmd = f"[ -f {model} ] && echo 'exists' || echo 'missing'"
            model_status = executor.run(check_model_cmd)

            if not model_status or 'missing' in model_status[0]:
                model_name = os.path.basename(os.path.dirname(model))
                print(f"   📦 Model '{model_name}' missing on worker {i+1}, downloading...")
                executor.run(f"cd {base_dir} && python3 launch.py {model_name} -skip-run -y")

            print(f"   ✅ Worker {i+1} prepared")

            worker_info.append({
                'executor': executor,
                'net': net,
                'app_port': app_port,
                'gpu_idx': gpu_idx,
                'base_dir': base_dir
            })

        # Phase 2: Start all workers simultaneously to minimize startup time difference
        print("\n🚀 Phase 2: Starting all workers simultaneously...")
        for i, info in enumerate(worker_info):
            executor = info['executor']
            base_dir = info['base_dir']
            gpu_idx = info['gpu_idx']

            # Kill any existing worker processes first
            executor.run("pkill -9 -f 'hpipe-worker' || true")

            # Use stdbuf to disable buffering for immediate log output
            worker_cmd = f"stdbuf -oL -eL ./hpipe-worker --port 9999 --nthreads {threads} --gpu-index {gpu_idx}"

            # Create a startup script to ensure proper logging
            startup_script = f"""#!/bin/bash
cd {base_dir}
exec 1>{base_dir}/worker.log
exec 2>&1
echo "=== Worker startup at $(date) ==="
echo "Command: {worker_cmd}"
echo "==="
{worker_cmd}
EXIT_CODE=$?
echo "=== Worker exited with code $EXIT_CODE at $(date) ==="
exit $EXIT_CODE
"""

            # Write startup script
            executor.run(f"cat > {base_dir}/start_worker.sh << 'EOFSCRIPT'\n{startup_script}\nEOFSCRIPT")
            executor.run(f"chmod +x {base_dir}/start_worker.sh")

            # Run worker via startup script (non-blocking)
            executor.run(f"nohup {base_dir}/start_worker.sh > /dev/null 2>&1 &", background=True)

            worker_endpoints.append(f"{info['net']['public_ip']}:{info['app_port']}")
            print(f"   Worker {i+1} started at {info['net']['public_ip']}:{info['app_port']} (GPU: {gpu_idx})")

        print("⏳ Giving all workers a moment to start up...")
        time.sleep(3)

        print("⏳ Waiting for all workers to be ready (listening on port 9999)...")
        ready_workers = [False] * len(self.executors)
        start_wait = time.time()
        timeout = 120 # Max 2 minutes to load and listen

        while not all(ready_workers) and (time.time() - start_wait < timeout):
            for i, executor in enumerate(self.executors):
                if ready_workers[i]:
                    continue

                # Check if process is still running
                proc_check = executor.run("pgrep -f 'hpipe-worker.*--port 9999'")
                if not proc_check:
                    print(f"   ❌ Worker {i+1} process not found! Checking logs...")
                    logs = executor.run("tail -50 /app/worker.log")
                    if logs:
                        print(f"   Last 50 lines of worker {i+1} log:")
                        for line in logs:
                            print(f"      {line}")
                    raise RuntimeError(f"Worker {i+1} process died or failed to start")

                # Check the last 10 lines of the log for the listening message
                logs = executor.run("tail -n 10 /app/worker.log")
                if logs and any("Listening on 0.0.0.0:9999" in line for line in logs):
                    print(f"   ✅ Worker {i+1} is ready and listening!")
                    ready_workers[i] = True
                elif logs:
                    # Show progress
                    last_line = logs[-1] if logs else ""
                    print(f"   ⏳ Worker {i+1}: {last_line[:80]}...")

            if not all(ready_workers):
                time.sleep(3)

        if not all(ready_workers):
            print("⚠️ Warning: Some workers did not report ready status within timeout.")
            print("   Collecting logs for debugging...")
            for i, executor in enumerate(self.executors):
                if not ready_workers[i]:
                    logs = executor.run("tail -50 /app/worker.log")
                    print(f"\n   Worker {i+1} log (last 50 lines):")
                    if logs:
                        for line in logs:
                            print(f"      {line}")
        else:
            print("✨ All workers are ready. Warming up for 2 seconds...")
            time.sleep(2)

        # 2. Start Root locally via script
        # Calculate absolute path to the script
        infra_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(infra_dir)
        script_path = os.path.join(project_root, "scripts", "run-hpipe-root.sh")

        print(f"👑 Running root node locally via {script_path}...")
        workers_arg = " ".join(worker_endpoints)
        
        # Collect device profiles from config
        device_profiles = []
        for w_cfg in worker_configs:
            profile = w_cfg.get('device_profile', {})
            tflops = profile.get('tflops', 100.0)
            bandwidth = profile.get('bandwidth', 50.0)
            latency = profile.get('latency', 0.0001)
            device_profiles.append(f"{tflops},{bandwidth},{latency}")

        device_profiles_arg = " ".join(device_profiles)

        script_cmd = [
            script_path,
            "--model", model,
            "--tokenizer", tokenizer,
            "--workers", workers_arg,
            "--nthreads", str(threads),
            "--steps", str(steps),
            "--prompt", prompt,
            "--device-profiles", device_profiles_arg
        ]
        
        try:
            # Execute from project root to ensure 'make' and binary execution works
            subprocess.run(script_cmd, check=True, cwd=project_root)
        except subprocess.CalledProcessError as e:
            print(f"❌ Root execution script failed: {e}")
        finally:
            self.collect_logs()

    def collect_logs(self):
        """Collects worker.log from each pod."""
        print("📊 Collecting worker logs...")
        log_dir = "infra/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        for i, executor in enumerate(self.executors):
            local_log_path = os.path.join(log_dir, f"worker_{i+1}.log")
            try:
                # Try absolute path first
                executor.download_file("/app/worker.log", local_log_path)
            except Exception as e:
                print(f"⚠️ Failed to collect log from worker {i+1}: {e}")

    def cleanup(self):
        print("🧹 Cleaning up cluster...")
        for i, pod in enumerate(self.worker_pods):
            w_cfg = self.worker_configs[i] if i < len(self.worker_configs) else {}
            # pod_id가 있으면 재사용 중이므로 stop만
            if 'pod_id' in w_cfg and w_cfg['pod_id']:
                pod.stop()
                print(f"   ⏸️  Stopped (reused pod): {pod.pod_id}")
            else:
                pod.terminate()
                print(f"   ❌ Terminated: {pod.pod_id}")