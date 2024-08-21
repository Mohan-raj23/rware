import numpy as np
import matplotlib.pyplot as plt
import os
from robotic_warehouse.warehouse5 import RewardType, Warehouse
from maddpg import MADDPG

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
        max_steps=500,
        reward_type=RewardType.INDIVIDUAL
    )
    num_agents = env.n_agents
    state_size = env.observation_space[0].shape[0]
    action_size = env.action_space[0].n
    return env, num_agents, state_size, action_size

def save_plot(episode_rewards, save_path='reward_plot_latest.png'):
    plt.figure()
    for i, rewards in enumerate(episode_rewards):
        plt.plot(rewards, label=f'Agent {i+1}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved at {save_path}")

def main():
    env, num_agents, state_dim, action_dim = load_env()
    action_bound = 1  # Assuming the action space is normalized between -1 and 1

    agent = MADDPG(
        num_agents=num_agents,
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=action_bound,
        actor_lr=0.01,
        critic_lr=0.01,
        gamma=0.95,
        tau=0.01,
        buffer_size=1000000,
        batch_size=64
    )

    num_episodes = 10000
    plot_save_interval = 10  # Save the plot every 100 episodes
    episode_rewards = [[] for _ in range(num_agents)]  # Store rewards for each agent

    if not os.path.exists('plots'):
        os.makedirs('plots')

    for episode in range(num_episodes):
        states = env.reset()
        episode_reward = [0 for _ in range(num_agents)]

        for step in range(env.max_steps):
            actions = agent.act(states)
            discrete_actions = [np.argmax(action) for action in actions]
            
            next_states, rewards, dones, _ = env.step(discrete_actions)
            # env.render()

            agent.remember(states, actions, rewards, next_states, dones)
            agent.train()

            states = next_states
            for i in range(num_agents):
                episode_reward[i] += rewards[i]

            if any(dones):
                break

        for i in range(num_agents):
            episode_rewards[i].append(episode_reward[i])

        if (episode + 1) % plot_save_interval == 0:
            save_path = 'plots/reward_plot_latest.png'
            save_plot(episode_rewards, save_path)

        print(f"Episode {episode} finished with rewards: {episode_reward}")

    # Save the final plot
    save_plot(episode_rewards, save_path='plots/reward_plot_latest.png')

if __name__ == "__main__":
    main()
