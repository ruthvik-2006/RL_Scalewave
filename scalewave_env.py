import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ScaleWaveEnv(gym.Env):
    """
    Gymnasium Environment for Edge Serverless Traffic Orchestration using RL.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, config):
        super(ScaleWaveEnv, self).__init__()
        self.config = config
        self.impls = config["equivalent_implementations"]
        self.num_impls = len(self.impls)
        self.concurrency_target = config["concurrency_target"]

        # Action Space: Continuous Dirichlet traffic split percentages [0.0, 1.0] sum to 1
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_impls,), dtype=np.float32
        )

        # Observation Space:
        # [In-flight Requests, CPU Util, GPU Util, Mem Util, Net Util, Prev Latency]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.in_flight_requests = 100.0
        self.resource_util = np.array([0.2, 0.1, 0.3, 0.15], dtype=np.float32)
        self.prev_latency = 50.0

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        norm_reqs = min(self.in_flight_requests / 1000.0, 1.0)
        norm_lat = min(self.prev_latency / 500.0, 1.0)
        return np.array([norm_reqs, *self.resource_util, norm_lat], dtype=np.float32)

    def step(self, action):
        # Normalize actions so weights sum to 1.0
        action_sum = np.sum(action)
        if action_sum <= 0:
            weights = np.ones(self.num_impls) / self.num_impls
        else:
            weights = action / action_sum

        # Simulate synthetic workload demand burst
        self.current_step += 1
        burst = np.sin(self.current_step / 3.0) * 400 + 500
        self.in_flight_requests = float(np.clip(burst + np.random.normal(0, 50), 50, 1200))

        # Calculate resource bottleneck pressures and latency
        total_latency = 0.0
        successful_reqs = 0.0
        cpu_load, gpu_load, net_load = 0.0, 0.0, 0.0

        for idx, w in enumerate(weights):
            allocated_reqs = self.in_flight_requests * w
            impl = self.impls[idx]

            # Replicas required based on concurrency target
            replicas_needed = np.ceil(allocated_reqs / self.concurrency_target)
            
            # Resource pressure calculations
            cpu_load += replicas_needed * impl["resource_costs"]["cpu"] / self.config["cluster_resources"]["cpu_cores"]
            gpu_load += replicas_needed * impl["resource_costs"]["gpu"] / self.config["cluster_resources"]["gpu_cores"]
            net_load += allocated_reqs * impl["resource_costs"]["network"] / self.config["cluster_resources"]["network_mbps"]

            # Bottleneck penalty multiplier
            penalty = 1.0
            if cpu_load > 0.95 or gpu_load > 0.95 or net_load > 0.95:
                penalty = 3.5

            lat = (impl["base_latency_ms"] + (allocated_reqs * 0.15)) * penalty
            total_latency += lat * w
            
            # Served requests capped by capacity and saturation
            max_capacity = replicas_needed * impl["max_throughput_per_replica"]
            successful_reqs += min(allocated_reqs, max_capacity / max(1.0, penalty - 0.5))

        self.resource_util = np.array([
            min(cpu_load, 1.0),
            min(gpu_load, 1.0),
            0.4,
            min(net_load, 1.0)
        ], dtype=np.float32)

        self.prev_latency = total_latency

        # Normalized Performance Metric = Throughput / Latency
        norm_performance = (successful_reqs / (total_latency / 1000.0 + 1e-5))

        # Reward formulation: Reward performance, penalize tail latencies and resource saturation
        reward = norm_performance / 50.0
        if cpu_load > 0.9 or gpu_load > 0.9:
            reward -= 2.0  # Saturation penalty

        terminated = self.current_step >= 200
        truncated = False

        info = {
            "successful_requests": successful_reqs,
            "latency_ms": total_latency,
            "norm_performance": norm_performance,
            "weights": weights
        }

        return self._get_obs(), reward, terminated, truncated, info