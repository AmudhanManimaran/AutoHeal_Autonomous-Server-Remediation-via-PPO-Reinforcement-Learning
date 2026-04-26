import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class SelfHealingEnv(gym.Env):
    """
    Custom 6D/7-Action Environment simulating a high-performance Kubernetes cluster.
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self, render_mode="console"):
        super(SelfHealingEnv, self).__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(7)

        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([100.0, 100.0, 2000.0, 1.0, 10000.0, 100.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.state = None
        self.max_steps = 200
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        self.state = np.array([
            random.uniform(20.0, 40.0),  
            random.uniform(30.0, 50.0), 
            random.uniform(50.0, 100.0),
            0.0,                        
            random.uniform(500, 1500),  
            random.uniform(10.0, 20.0)   
        ], dtype=np.float32)

        return self.state, {}

    def step(self, action):
        self.current_step += 1
        cpu, mem, latency, err_rate, rps, thread_q = self.state

        if action == 1:
            mem = max(20.0, mem - 40.0)
            thread_q = max(0.0, thread_q - 50.0)
        elif action == 2: 
            cpu = max(10.0, cpu - 40.0)
            rps = max(500.0, rps - 2000.0) 
        elif action == 3: 
            cpu = max(20.0, cpu - 30.0)
            latency = max(50.0, latency - 150.0)
        elif action == 4: 
            rps = max(100.0, rps - 4000.0)
            cpu = max(10.0, cpu - 50.0)
        elif action == 5: 
            err_rate = max(0.0, err_rate - 0.4)
            thread_q = max(0.0, thread_q - 60.0)
        elif action == 6:
            latency = max(50.0, latency - 300.0)
            err_rate = 0.0

        if random.random() < 0.1:
            rps += random.uniform(2000, 5000)
     
        rps += random.uniform(-200, 300)
        cpu += random.uniform(-5.0, 10.0) + (rps / 1000.0) 
        mem += random.uniform(-2.0, 8.0)
        thread_q += random.uniform(-2.0, 12.0) + (cpu / 20.0) 

        if thread_q > 70.0:
            latency += random.uniform(100.0, 400.0)
        else:
            latency += random.uniform(-10.0, 20.0)

        if cpu > 85.0 or mem > 85.0 or thread_q > 85.0:
            err_rate += random.uniform(0.1, 0.3)

        cpu = np.clip(cpu, 0.0, 100.0)
        mem = np.clip(mem, 0.0, 100.0)
        latency = np.clip(latency, 0.0, 2000.0)
        err_rate = np.clip(err_rate, 0.0, 1.0)
        rps = np.clip(rps, 0.0, 10000.0)
        thread_q = np.clip(thread_q, 0.0, 100.0)

        self.state = np.array([cpu, mem, latency, err_rate, rps, thread_q], dtype=np.float32)

       
        reward = 0.0
        terminated = False

        if cpu >= 95.0 or mem >= 95.0 or err_rate >= 0.8 or thread_q >= 95.0:
            reward = -500.0  
            terminated = True
        else:
            is_degrading = cpu >= 75.0 or mem >= 75.0 or latency >= 300.0 or err_rate >= 0.2 or thread_q >= 70.0
            
            if is_degrading:
                if action == 0: 
                    reward -= 50.0  
                else:
                    
                    reward += 50.0  
            else:
                if action == 0:
                    reward += 1.0   
                else:
                   
                    reward -= 20.0  
        
        truncated = bool(self.current_step >= self.max_steps)

        return self.state, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "console":
            cpu, mem, lat, err, rps, tq = self.state
            print(f"Step: {self.current_step} | CPU: {cpu:.1f}% | Mem: {mem:.1f}% | Lat: {lat:.1f}ms | Err: {err:.2f} | RPS: {rps:.0f} | TQ: {tq:.1f}%")