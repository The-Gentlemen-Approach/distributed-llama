import os
import json
import argparse
from dotenv import load_dotenv
from cluster_orchestrator import DistributedLlamaOrchestrator

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Distributed Llama RunPod Orchestrator")
    parser.add_argument("--config", type=str, default="infra/cluster_config.json", help="Path to cluster config JSON")
    parser.add_argument("--ssh_key", type=str, default="~/.ssh/id_rsa", help="Path to private SSH key")
    parser.add_argument("--keep", action="store_true", help="Keep pods running after execution")

    args = parser.parse_args()

    # Configuration
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("❌ Error: Missing RUNPOD_API_KEY")
        return

    import runpod
    runpod.api_key = api_key

    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found at {args.config}")
        return

    config = load_config(args.config)
    orchestrator = DistributedLlamaOrchestrator(ssh_key=args.ssh_key)

    try:
        orchestrator.setup_cluster(config)
        orchestrator.run_inference(config)
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if not args.keep:
            orchestrator.cleanup()
        else:
            print("ℹ️ Cluster preserved.")

if __name__ == "__main__":
    main()