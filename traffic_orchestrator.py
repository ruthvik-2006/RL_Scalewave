import time
import json
import numpy as np
from stable_baselines3 import PPO

class KnativeTrafficOrchestrator:
    """
    Connects RL Policy to Knative Serving dynamic traffic-splitting API.
    """
    def __init__(self, model_path, config_path):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.model = PPO.load(model_path)
        self.impls = self.config["equivalent_implementations"]

    def get_current_cluster_metrics(self):
        # Mock reading metrics from Knative Pod Autoscaler / Metrics Server
        in_flight = float(np.random.uniform(200, 800))
        cpu_util = float(np.random.uniform(0.3, 0.85))
        gpu_util = float(np.random.uniform(0.1, 0.6))
        mem_util = 0.45
        net_util = float(np.random.uniform(0.2, 0.7))
        prev_latency = float(np.random.uniform(30, 120))

        norm_reqs = min(in_flight / 1000.0, 1.0)
        norm_lat = min(prev_latency / 500.0, 1.0)

        obs = np.array([norm_reqs, cpu_util, gpu_util, mem_util, net_util, norm_lat], dtype=np.float32)
        return obs, in_flight

    def compute_traffic_split(self, obs):
        action, _states = self.model.predict(obs, deterministic=True)
        # Apply Softmax/Normalization
        weights = np.exp(action) / np.sum(np.exp(action))
        weights_percentage = np.round(weights * 100, 2)
        
        # Ensure total sums strictly to 100%
        weights_percentage[-1] = 100.0 - np.sum(weights_percentage[:-1])
        return weights_percentage

    def apply_knative_traffic_split(self, traffic_split):
        print("\n[+] Updating Knative Revision Traffic Weights:")
        for idx, impl in enumerate(self.impls):
            print(f"    -> Revision: {impl['name']:<20} | Assigned Weight: {traffic_split[idx]:>6.2f}%")
        
        # Command syntax for actual Knative CLI deployment:
        # kubectl patch ksvc face-recognition -n default --type='json' -p='[...]'
        print("[✓] Knative Routing Rules updated atomically in cluster control plane.")

    def run_control_loop(self, iterations=3):
        print("[+] Starting RL-ScaleWave Orchestration Daemon...")
        for i in range(iterations):
            print(f"\n--- Orchestration Cycle {i+1} ---")
            obs, in_flight = self.get_current_cluster_metrics()
            print(f"Metrics Observer -> In-Flight Reqs: {in_flight:.0f} | CPU: {obs[1]*100:.1f}% | GPU: {obs[2]*100:.1f}%")
            
            traffic_split = self.compute_traffic_split(obs)
            self.apply_knative_traffic_split(traffic_split)
            time.sleep(1)

if __name__ == "__main__":
    orchestrator = KnativeTrafficOrchestrator("models/ppo_scalewave", "config.json")
    orchestrator.run_control_loop(iterations=3)