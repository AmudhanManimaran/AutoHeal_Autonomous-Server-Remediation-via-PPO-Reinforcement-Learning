import time
import os
import csv
from stable_baselines3 import PPO
from envs.self_healing_env import SelfHealingEnv

# --- UPDATED 7-ACTION SPACE ---
action_names = {
    0: "A0: NO-OP (MONITOR)",
    1: "A1: GRACEFUL POD EVICTION",
    2: "A2: HORIZONTAL POD AUTOSCALING (HPA)",
    3: "A3: VERTICAL POD AUTOSCALING (VPA)",
    4: "A4: LOAD SHEDDING",
    5: "A5: CIRCUIT BREAKING",
    6: "A6: TRAFFIC SHIFTING (CANARY)"
}

print("Loading 6D/7-Action Environment and Trained Agent...")
env = SelfHealingEnv(render_mode="console")

# Load the NEW 6D model
model_path = "models/ppo_self_healing_agent_final"
try:
    agent = PPO.load(model_path)
    print(f"Successfully loaded model: {model_path}")
except:
    print(f"Could not find {model_path}. Did train_agent.py finish running?")
    exit()

obs, info = env.reset()
print("\n--- STARTING LIVE DEMONSTRATION & DATA LOGGING ---")

# Setup CSV Logging for IEEE Paper Graphs
log_file = "simulation_telemetry.csv"
with open(log_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write CSV Header
    writer.writerow(["Step", "CPU(%)", "Memory(%)", "Latency(ms)", "ErrorRate", "RPS", "ThreadQueue(%)", "Agent_Action", "Reward"])

    for i in range(50): # 50 steps to capture a full flash crowd event
        env.render()
        
        # Agent predicts the best action based on the 6D telemetry
        action, _states = agent.predict(obs, deterministic=True)
        action_idx = int(action)
        
        print(f">> Agent Decision: {action_names[action_idx]}")
        
        # Step the environment forward
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Log telemetry to CSV
        cpu, mem, lat, err, rps, tq = obs
        writer.writerow([i, cpu, mem, lat, err, rps, tq, action_names[action_idx], reward])
        
        time.sleep(0.5) 
        print("-" * 60)
        
        if terminated or truncated:
            print("Episode Ended (System Crashed or Reached Max Steps). Resetting...")
            obs, info = env.reset()

print(f"Demonstration Complete. Telemetry saved to '{log_file}' for Graph Generation.")