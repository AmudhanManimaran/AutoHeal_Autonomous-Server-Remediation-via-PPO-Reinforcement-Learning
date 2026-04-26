import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from envs.self_healing_env import SelfHealingEnv


if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("logs"):
    os.makedirs("logs")

print("Initializing the 6D/7-Action Training Environment...")

train_env = Monitor(SelfHealingEnv())
check_env(train_env, warn=True)

print("Initializing the Validation Environment...")
eval_env = Monitor(SelfHealingEnv())


eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path='./models/best_model/',
    log_path='./logs/', 
    eval_freq=10000, 
    n_eval_episodes=5,
    deterministic=True, 
    render=False
)

policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))

print("Initializing the PPO Agent with Paper Hyperparameters...")
agent = PPO(
    "MlpPolicy", 
    train_env, 
    verbose=1, 
    learning_rate=0.0003, 
    n_steps=2048,
    batch_size=64,           
    clip_range=0.2,          
    gamma=0.99,              
    ent_coef=0.05, 
    policy_kwargs=policy_kwargs,
    tensorboard_log="./logs/ppo_tensorboard/"
)

print("Starting Training Loop (300,000 Timesteps) with Periodic Validation...")

agent.learn(total_timesteps=300000, callback=eval_callback)

print("Training Complete. Saving the final model...")
agent.save("models/ppo_self_healing_agent_final")
print("Model saved successfully. Check the '/logs/' folder for validation data!")