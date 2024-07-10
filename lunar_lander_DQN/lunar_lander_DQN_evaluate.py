import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

env = gym.make(
    "LunarLander-v2",
    render_mode="human",
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
)

run_device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Running on {run_device}")

TEST_NUM_EPISODES = 10
MODEL_PATH = "models/lunar_lander_dqn_model.pth"

model = torch.load(MODEL_PATH)
model.eval()

for episode in range(TEST_NUM_EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        env.render()
        state = torch.tensor(state, device=run_device, dtype=torch.float32)
        action = model(state).argmax().item()
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print(f"Episode {episode + 1} finished with reward {total_reward}")
