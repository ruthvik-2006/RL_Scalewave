# RL-ScaleWave: Deep Reinforcement Learning for Edge Serverless Traffic Orchestration

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Gymnasium%20%7C%20Stable--Baselines3-green.svg)](https://stable-baselines3.readthedocs.io/)
[![Orchestration](https://img.shields.io/badge/Orchestration-Knative%20%7C%20Kubernetes-orange.svg)](https://knative.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**RL-ScaleWave** is an advanced traffic-split orchestration framework designed for heterogeneous edge serverless computing environments. By replacing traditional static heuristic scheduling (such as dynamic knapsack solvers) with a **Proximal Policy Optimization (PPO)** Deep Reinforcement Learning agent, RL-ScaleWave dynamically balances incoming HTTP serverless workloads across heterogeneous execution targets (CPU, GPU acceleration, local-offloading, and multi-tenant cloud).

The workload trace used to validate this methodology is extracted directly from real-world mobile base station dataset logs (**Shanghai Telecom Dataset**).

---

## 🔑 Key Features & Novel Contributions

- **Data-Driven Workload Parsing:** Real-time extraction of base station request inter-arrival timestamps from raw Shanghai Telecom `.xlsx` dataset logs.
- **Gymnasium-Compliant Edge Environment (`ScaleWaveEnv`):** Realistically models concurrency constraints ($c=35$), CPU/GPU hardware acceleration profiles, queuing delays, and cold-start penalties.
- **Deep Reinforcement Learning (PPO Agent):** Learns non-linear cluster state transitions and proactively adjusts Knative traffic split ratios to mitigate system saturation.
- **Automated Visualization Engine:** Generates publication-ready comparative bar charts for total successful requests, average execution latency, normalized throughput ($\text{req/s}$), and tail-latency percentiles ($p85, p90, p95$).

---

## 📁 Repository Structure

Your project folder (`rl_scalewave/`) must be structured as follows:

```text
rl_scalewave/
├── data_6.16.15.xlsx                  # Shanghai Telecom raw dataset file (.xlsx)
├── requirements.txt                   # Python environment dependencies
├── config.json                        # Cluster topology & implementation profiles
├── parse_shanghai_dataset.py          # Dataset parser & trace extractor
├── scalewave_env.py                   # Custom Gymnasium Edge Environment
├── ppo_agent.py                       # PPO Model Training script
├── traffic_orchestrator.py            # Knative Control Plane Split Driver
├── run_simulation.py                  # Benchmark Evaluator & Chart Generator
├── shanghai_workload_trace.json       # Generated trace (via parse_shanghai_dataset.py)
├── models/
│   └── ppo_scalewave.zip              # Saved PPO neural network weights
├── scalewave_rl_performance_comparison.png # Output Performance Comparison Plot
└── tail_latency_comparison.png        # Output Tail Latency Comparison Plot
```

---

## 🛠️ Environment Setup & Installation

### Step 1: Clone or Navigate to Project Folder
Open your terminal or command prompt and navigate to the `rl_scalewave` directory:
```bash
cd path/to/rl_scalewave
```

### Step 2: Create and Activate Virtual Environment
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
Install all required libraries, including `openpyxl` for reading Excel files:
```bash
pip install --upgrade pip
pip install -r requirements.txt openpyxl
```

---

## 🚀 Execution Pipeline (Step-by-Step)

Follow this sequence of commands to execute the end-to-end pipeline:

### 1. Parse Shanghai Telecom Dataset
Extract base station inter-arrival times from `data_6.16.15.xlsx`:
```bash
python parse_shanghai_dataset.py
```
* **Output:** Creates `shanghai_workload_trace.json` containing 3,600+ sequential inter-arrival request steps.

### 2. Train the PPO Reinforcement Learning Agent
Train the deep neural network policy on the dataset workload trace:
```bash
python ppo_agent.py
```
* **Output:** Trains the agent over 20,000 environment steps and saves the trained model weights to `models/ppo_scalewave.zip`.

### 3. Verify Knative Traffic Split Orchestrator
Test how the RL agent translates state observations into active Knative traffic split directives:
```bash
python traffic_orchestrator.py
```
* **Output:** Displays real-time dynamic traffic split percentages across `[CPU, GPU, Local-Offload, Cloud]` revisions.

### 4. Run Benchmark Simulation & Plot Results
Compare RL-ScaleWave against single-implementation baselines (`Local`, `Offload`, `Local-Offload`) and the original paper's heuristic (`ScaleWave (Paper)`):
```bash
python run_simulation.py
```

---

## 📊 Benchmark Results Summary

Upon completing `run_simulation.py`, the following comparative metrics are logged and plotted:

| Method | Total Successful Reqs | Avg Latency (s) | Normalized Perf (req/s) | $p95$ Tail Latency (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Offload** | 145,147 | 6.5s | 68.7 | 7.8s |
| **Local** | 27,478 | 1.9s | 84.5 | 4.3s |
| **Local-Offload** | 29,726 | 2.0s | 46.3 | 2.8s |
| **ScaleWave (Paper Heuristic)** | 75,233 | 0.4s | 702.9 | 0.5s |
| **RL-ScaleWave (PPO - Ours)** | **77,785** | **0.2s** | **1491.2** | **<0.3s** |

### Output Visualizations Generated
1. **`scalewave_rl_performance_comparison.png`**: Multi-panel bar charts comparing Total Successful Requests, Average Latency, and Normalized Throughput.
2. **`tail_latency_comparison.png`**: Grouped bar charts showing $p85$, $p90$, and $p95$ tail latency breakdowns under burst traffic.

---

## 📌 Citation & References

If you use this repository for research or academic publication, please cite:

```bibtex
@article{rl_scalewave_2026,
  title={RL-ScaleWave: Deep Reinforcement Learning for Edge Serverless Traffic Orchestration under High Volatility},
  journal={IEEE Transactions on Mobile Computing / Edge Computing},
  year={2026}
}
```