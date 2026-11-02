import os
import json
from stable_baselines3 import PPO
from scalewave_env import ScaleWaveEnv

def train_ppo_agent(config_path="config.json", model_save_path="models/ppo_scalewave"):
    with open(config_path, "r") as f:
        config = json.load(f)

    env = ScaleWaveEnv(config)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    print("[+] Initializing PPO Agent for ScaleWave Optimization...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )

    print("[+] Training PPO RL Agent across 20,000 Edge workload steps...")
    model.learn(total_timesteps=20000)
    
    model.save(model_save_path)
    print(f"[✓] Model trained successfully and saved to '{model_save_path}.zip'")
    return model

if __name__ == "__main__":
    train_ppo_agent()