import runpod
import time
from typing import Optional, Dict, Any

class RunPodPod:
    """Responsibility 1: Manage RunPod lifecycle (start, stop, terminate, info)."""
    def __init__(self, pod_id: str):
        self.pod_id = pod_id
        self._pod_data: Dict[str, Any] = {}

    @classmethod
    def create(cls, name: str, template_id: str, gpu_type_id: str) -> 'RunPodPod':
        pod = runpod.create_pod(
            name=name,
            template_id=template_id,
            gpu_type_id=gpu_type_id,
            cloud_type="COMMUNITY",
            min_download=8000,
            min_upload=8000
        )
        return cls(pod['id'])

    def refresh(self):
        try:
            self._pod_data = runpod.get_pod(self.pod_id)
            
            if not self._pod_data:
                all_pods = runpod.get_pods()
                for p in all_pods:
                    if p['id'] == self.pod_id:
                        self._pod_data = p
                        print(f"🔍 [DEBUG] Found in get_pods list: {self._pod_data}")
                        break
        except Exception as e:
            print(f"⚠️ Error refreshing pod data: {e}")
            self._pod_data = None

    def terminate(self):
        runpod.terminate_pod(self.pod_id)

    def stop(self):
        runpod.stop_pod(self.pod_id)

    def resume(self):
        runpod.resume_pod(self.pod_id)

    @property
    def status(self) -> str:
        self.refresh()
        if not self._pod_data:
            return 'UNKNOWN'
        
        # 전문가의 조언대로 최상위 desiredStatus를 사용합니다.
        return str(self._pod_data.get('desiredStatus', 'UNKNOWN')).upper()

    @property
    def is_running(self) -> bool:
        # RUNNING 상태이면서 동시에 네트워크 정보(IP)가 할당되었는지 확인합니다.
        if self.status != 'RUNNING':
            return False
            
        info = self.network_info
        return info.get("public_ip") is not None

    @property
    def network_info(self) -> Dict[str, Any]:
        """Extracts IPs and all mapped ports based on the recommended structure."""
        if not self._pod_data:
            self.refresh()
        
        info = {"public_ip": None, "internal_ip": None, "ports": []}
        if not self._pod_data:
            return info

        # publicIp 필드 또는 address 필드 확인
        info["public_ip"] = self._pod_data.get('publicIp') or self._pod_data.get('address')
        
        runtime = self._pod_data.get('runtime') or {}
        # 실제 응답에서 확인된 리스트 형태의 ports 처리
        ports_data = runtime.get('ports', [])
        
        if isinstance(ports_data, list):
            for p in ports_data:
                # 내부 포트(privatePort) -> 외부 포트(publicPort) 매핑 저장
                info["ports"].append({
                    "containerPort": p.get('privatePort'),
                    "publicPort": p.get('publicPort')
                })
                # 만약 public_ip가 아직 없다면 여기서 추출 시도
                if not info["public_ip"] and p.get('isIpPublic'):
                    info["public_ip"] = p.get('ip')
        
        return info

    def wait_for_status(self, target_status: str, timeout: int = 300):
        target_status = target_status.upper()
        start_time = time.time()
        print(f"⏳ Pod ({self.pod_id}) 시작 대기 중...", end="", flush=True)
        
        while time.time() - start_time < timeout:
            current_status = self.status
            
            # RUNNING 상태가 되면 네트워크 정보까지 확인하여 최종 완료 판단
            if current_status == target_status:
                if target_status != 'RUNNING' or self.is_running:
                    print(f"\n✅ Pod 실행 완료! ({current_status})")
                    return True
            
            if current_status in ["EXITED", "FAILED"]:
                print(f"\n❌ Pod가 시작 도중 종료되었습니다. (Status: {current_status})")
                return False
                
            print(".", end="", flush=True)
            time.sleep(3) # 전문가의 조언대로 3초 간격 폴링
            
        print("\n❌ 시간 초과 (Timeout)")
        raise TimeoutError(f"Pod {self.pod_id} did not reach {target_status} within {timeout}s")
