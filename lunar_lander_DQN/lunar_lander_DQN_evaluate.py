import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

env = gym.make(
    "LunarLander-v2",
    render_mode="human",
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
)

class DQN(nn.Module):
  def __init__(self, n_observations, n_actions):
    super(DQN, self).__init__()
    self.fc1 = nn.Linear(n_observations, 128)
    self.fc2 = nn.Linear(128, 128)
    self.fc3 = nn.Linear(128, n_actions)
  
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
  
run_device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Running on {run_device}")

TEST_NUM_EPISODES = 10
MODEL_PATH = "models/lunar_lander_dqn_model.pth"

model = DQN(env.observation_space.shape[0], env.action_space.n)
model = model.load_state_dict(torch.load(MODEL_PATH))
# model.eval()

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
