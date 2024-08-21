import copy
import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from robotic_warehouse.warehouse import RewardType, Warehouse  # Ensure the package is correctly imported
from tqdm import tqdm

# Check if a GPU is available and use it
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_size)

    def forward(self, x):
        x = th.relu(self.fc1(x))
        x = th.relu(self.fc2(x))
        x = th.relu(self.fc3(x))
        x = th.relu(self.fc4(x))
        x = th.relu(self.fc5(x))
        return self.out(x)

# Define the Q-learning agent
class QLearningAgent:
    def __init__(self, state_size, action_size, lr=0.0001, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_network = QNetwork(state_size, action_size).to(device)
        self.target_q_network = copy.deepcopy(self.q_network).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        state = th.FloatTensor(state).to(device).unsqueeze(0)
        with th.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def train(self, state, action, reward, next_state, done):
        state = th.FloatTensor(state).to(device).unsqueeze(0)
        next_state = th.FloatTensor(next_state).to(device).unsqueeze(0)
        reward = th.FloatTensor([reward]).to(device)
        action = th.LongTensor([action]).to(device)
        done = th.FloatTensor([done]).to(device)

        q_values = self.q_network(state)
        next_q_values = self.target_q_network(next_state)
        q_value = q_values[0, action]
        next_q_value = reward + (1 - done) * self.gamma * next_q_values.max()

        loss = self.loss_fn(q_value, next_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

def load_env():
    env = Warehouse(
        shelf_columns=3,
        column_height=8,
        shelf_rows=1,
        n_agents=2,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=2,
        max_inactivity_steps=None,
        max_steps=1000,
        reward_type=RewardType.INDIVIDUAL
    )
    num_agents = env.n_agents
    state_size = env.observation_space[0].shape[0]
    action_size = env.action_space[0].n
    return env, num_agents, state_size, action_size

def plot_stats(reward_history, filename_prefix, window_size=50):
    plt.figure(figsize=(10, 5))
    for i in reward_history:
        episodes = list(range(window_size, len(reward_history[i]) + 1, window_size))
        smoothed_rewards = [
            np.mean(reward_history[i][j-window_size:j])
            for j in episodes
        ]
        plt.plot(episodes, smoothed_rewards, label=f"Agent {i} (Reward)", linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward vs Episode (All Agents)')
    plt.legend()
    plt.savefig(f'{filename_prefix}_rewards_vs_episode_combined.png')
    plt.show()

def main():
    env, num_agents, state_size, action_size = load_env()
    agents = [QLearningAgent(state_size, action_size) for _ in range(num_agents)]
    reward_history = {i: [] for i in range(num_agents)}

    for episode in tqdm(range(12000), desc="Training Progress"):
        state = env.reset()
        total_rewards = np.zeros(num_agents)

        for t in range(600):
            actions = []
            for i in range(num_agents):
                action = agents[i].select_action(state[i])
                actions.append(action)
            next_state, reward, done, _ = env.step(actions)
            env.render()
            for i in range(num_agents):
                agents[i].train(state[i], actions[i], reward[i], next_state[i], done[i])
                total_rewards[i] += reward[i]

            state = next_state

            if any(done):
                break

        for agent in agents:
            agent.update_target_network()

        for i in range(num_agents):
            reward_history[i].append(total_rewards[i])

        if episode % 10 == 0:
            tqdm.write(f"Episode {episode} completed")

    for i in range(num_agents):
        th.save(agents[i].q_network.state_dict(), f'q_learning_agent_{i}.pth')

    plot_stats(reward_history, 'training')

if __name__ == "__main__":
    main()
