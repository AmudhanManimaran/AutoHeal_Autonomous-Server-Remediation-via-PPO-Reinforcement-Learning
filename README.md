# AeroHeal — Autonomous Server Remediation via Proximal Policy Optimization (PPO)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=flat-square&logo=pytorch)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.2.1-purple?style=flat-square)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-green?style=flat-square)
![TensorBoard](https://img.shields.io/badge/TensorBoard-Logging-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> A reinforcement learning system that trains a **PPO agent** on a custom Gymnasium environment simulating a Kubernetes-style server cluster — learning to autonomously detect degradation and execute targeted remediation actions before system failure, across 300,000 training timesteps.

---

## 🎯 Key Features

- **Custom 6D Gymnasium environment** — continuous server telemetry state space
- **7-action remediation policy** — Pod Eviction, HPA, VPA, Load Shedding, Circuit Breaking, Traffic Shifting, NO-OP
- **PPO agent** (Stable-Baselines3) with dual-head MLP policy (pi/vf: [64, 64])
- **EvalCallback** — periodic validation every 10,000 steps, best model checkpointing
- **TensorBoard training logs** — full PPO_1 event files included
- **Benchmark comparison** — vs Static Heuristics (HPA) and Legacy DRL (DQN)
- **Telemetry CSV logging** — 50-step live simulation with per-step action recording
- **3 result figures** — convergence curve, traffic mitigation, MTTR bar chart

---

## 📊 Benchmark Results

| Metric | Static Heuristics (HPA) | Legacy DRL (DQN) | AeroHeal (PPO) |
|--------|------------------------|-------------------|----------------|
| Mean Time To Recovery (MTTR) | 145.2s | 86.4s | **12.8s** |
| SLA Violation Rate | 18.2% | 9.7% | **~1.6%** |
| Policy Stability | Rule-based | Unstable (catastrophic drops) | **Monotonic convergence** |
| Catastrophic Crash Rate | High | Moderate | **Near-Zero** |
| Actuation Strategy | Reactive (scaling only) | Reactive (limited scope) | **Proactive (7-action)** |

**91.1% faster recovery** vs. Static Heuristics | **91.2% reduction** in SLA violations vs. Static Heuristics

---

## 🏗️ System Architecture

```
Server Telemetry State (6D observation)
  ┌────────────────────────────────────────┐
  │  CPU(%)  Mem(%)  Latency(ms)          │
  │  ErrorRate  RPS  ThreadQueue(%)        │
  └────────────────────────────────────────┘
                    │
                    ▼
     ┌──────────────────────────────┐
     │     PPO Agent (MlpPolicy)    │
     │   pi:  [64, 64] — Actor      │
     │   vf:  [64, 64] — Critic     │
     │   Activation: Tanh           │
     └──────────────────────────────┘
                    │
                    ▼
     7-Action Remediation Decision
  ┌────────────────────────────────────────┐
  │  A0: NO-OP (Monitor)                  │
  │  A1: Graceful Pod Eviction            │
  │  A2: Horizontal Pod Autoscaling (HPA) │
  │  A3: Vertical Pod Autoscaling (VPA)   │
  │  A4: Load Shedding                    │
  │  A5: Circuit Breaking                 │
  │  A6: Traffic Shifting (Canary)        │
  └────────────────────────────────────────┘
                    │
                    ▼
        Environment Step + Reward Signal
  ┌────────────────────────────────────────┐
  │  Degrading + Action → +50             │
  │  Degrading + NO-OP  → -50             │
  │  Stable  + NO-OP    → +1              │
  │  Stable  + Action   → -20             │
  │  Crash (CPU/Mem/Err > threshold) → -500│
  └────────────────────────────────────────┘
```

---

## 🧠 Environment Details

### State Space (6D Continuous)
| Dimension | Feature | Range |
|-----------|---------|-------|
| 0 | CPU Utilization (%) | 0–100 |
| 1 | Memory Utilization (%) | 0–100 |
| 2 | Request Latency (ms) | 0–2000 |
| 3 | Error Rate | 0–1 |
| 4 | Requests Per Second (RPS) | 0–10,000 |
| 5 | Thread Queue Depth (%) | 0–100 |

### Stochastic Dynamics
- 10% probability of random flash crowd spike (RPS +2,000–5,000)
- Continuous Gaussian noise on CPU, Memory, RPS
- Thread queue depth drives latency spikes when > 70%
- Error rate accumulates when CPU/Memory/Thread Queue exceed 85%

### Termination Conditions
- CPU ≥ 95% OR Memory ≥ 95% OR Error Rate ≥ 0.8 OR Thread Queue ≥ 95% → **Crash** (reward: -500)
- Max 200 steps per episode → **Truncation**

---

## 🏋️ Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Total Timesteps | 300,000 |
| Learning Rate | 0.0003 |
| n_steps | 2048 |
| Batch Size | 64 |
| Clip Range | 0.2 |
| Gamma | 0.99 |
| Entropy Coefficient | 0.05 |
| Policy Architecture | MLP [64, 64] (pi + vf) |
| Activation | Tanh |
| Eval Frequency | Every 10,000 steps |
| Eval Episodes | 5 |

---

## 📁 Project Structure

```
AeroHeal/
│
├── envs/
│   ├── self_healing_env.py         # Custom Gymnasium environment (6D/7-Action)
│   └── __init__.py
│
├── models/
│   ├── ppo_self_healing_agent_final.zip   # Final trained PPO model
│   └── best_model/
│       └── best_model.zip                 # Best checkpoint (EvalCallback)
│
├── logs/
│   ├── evaluations.npz                    # Periodic evaluation results
│   └── ppo_tensorboard/PPO_1/             # TensorBoard event files
│
├── train_agent.py                  # PPO training script
├── test_agent.py                   # Live simulation + telemetry logging
├── generate_benchmarks.py          # MTTR + SLA metrics from telemetry
├── generate_graphs.py              # Convergence, mitigation, MTTR figures
│
├── simulation_telemetry.csv        # 50-step simulation log
├── aeroheal_benchmarks.csv         # Benchmark comparison table
├── fig_convergence.png             # Training convergence: PPO vs DQN
├── fig_mitigation.png              # Flash crowd mitigation visualization
├── fig_mttr.png                    # MTTR benchmark bar chart
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/AmudhanManimaran/AeroHeal.git
cd AeroHeal
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train the PPO Agent
```bash
python train_agent.py
```
Trains for 300,000 timesteps. Saves:
- `models/ppo_self_healing_agent_final.zip`
- `models/best_model/best_model.zip` (best checkpoint)
- TensorBoard logs to `logs/ppo_tensorboard/`

### Monitor Training (TensorBoard)
```bash
tensorboard --logdir logs/ppo_tensorboard/
```

### Run Live Simulation
```bash
python test_agent.py
```
Runs 50-step live simulation, prints per-step agent decisions, saves `simulation_telemetry.csv`.

### Generate Benchmark Results
```bash
python generate_benchmarks.py
```
Computes MTTR and SLA violation rate from telemetry, saves `aeroheal_benchmarks.csv`.

### Generate Result Figures
```bash
python generate_graphs.py
```
Saves:
- `fig_convergence.png` — PPO vs DQN training convergence
- `fig_mitigation.png` — Flash crowd latency mitigation
- `fig_mttr.png` — MTTR benchmark comparison

---

## 📈 Result Figures

### Training Convergence
![Convergence](fig_convergence.png)

### Flash Crowd Mitigation
![Mitigation](fig_mitigation.png)

### MTTR Benchmark
![MTTR](fig_mttr.png)

---

## 📦 Requirements

```
torch>=2.0.0
gymnasium==0.29.1
stable-baselines3==2.2.1
numpy==1.26.4
pandas>=1.3.0
matplotlib>=3.5.0
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Amudhan Manimaran**
- 🌐 Portfolio: [amudhanmanimaran.github.io/Portfolio](https://amudhanmanimaran.github.io/Portfolio/)
- 💼 LinkedIn: [linkedin.com/in/amudhan-manimaran-3621bb32a](https://www.linkedin.com/in/amudhan-manimaran-3621bb32a)
- 🐙 GitHub: [github.com/AmudhanManimaran](https://github.com/AmudhanManimaran)
