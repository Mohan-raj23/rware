import numpy as np
import tensorflow as tf
from collections import deque
import random

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, action_bound, name='actor'):
        super(Actor, self).__init__(name=name)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.out = tf.keras.layers.Dense(action_dim, activation='tanh')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        action = self.out(x) * self.action_bound
        return action

class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim, name='critic'):
        super(Critic, self).__init__(name=name)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.out = tf.keras.layers.Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], axis=1)
        x = self.fc1(x)
        x = self.fc2(x)
        q_value = self.out(x)
        return q_value

class ReplayBuffer:
    def __init__(self, max_size, batch_size):
        self.buffer = deque(maxlen=max_size)
        self.batch_size = batch_size

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def size(self):
        return len(self.buffer)

class MADDPG:
    def __init__(self, num_agents, state_dim, action_dim, action_bound, actor_lr, critic_lr, gamma, tau, buffer_size, batch_size):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.actors = [Actor(state_dim, action_dim, action_bound, name=f'actor_{i}') for i in range(num_agents)]
        self.critics = [Critic(state_dim * num_agents, action_dim * num_agents, name=f'critic_{i}') for i in range(num_agents)]
        self.target_actors = [Actor(state_dim, action_dim, action_bound, name=f'target_actor_{i}') for i in range(num_agents)]
        self.target_critics = [Critic(state_dim * num_agents, action_dim * num_agents, name=f'target_critic_{i}') for i in range(num_agents)]

        self.actor_optimizers = [tf.keras.optimizers.Adam(actor_lr) for _ in range(num_agents)]
        self.critic_optimizers = [tf.keras.optimizers.Adam(critic_lr) for _ in range(num_agents)]

        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)

        self._initialize_target_networks()

    def _initialize_target_networks(self):
        for i in range(self.num_agents):
            self.target_actors[i].set_weights(self.actors[i].get_weights())
            self.target_critics[i].set_weights(self.critics[i].get_weights())

    def update_target_networks(self):
        for i in range(self.num_agents):
            self._soft_update(self.target_actors[i], self.actors[i])
            self._soft_update(self.target_critics[i], self.critics[i])

    def _soft_update(self, target, source):
        for target_param, param in zip(target.trainable_variables, source.trainable_variables):
            target_param.assign(self.tau * param + (1.0 - self.tau) * target_param)

    def act(self, states):
        actions = []
        for i, actor in enumerate(self.actors):
            state = np.expand_dims(states[i], axis=0).astype(np.float32)
            action = actor(state)
            actions.append(action[0].numpy())
        return actions

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))

    def train(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # # Debugging prints for dimensions
        # print(f"States shape: {states.shape}")
        # print(f"Actions shape: {actions.shape}")
        # print(f"Rewards shape: {rewards.shape}")
        # print(f"Next States shape: {next_states.shape}")
        # print(f"Dones shape: {dones.shape}")

        for i in range(self.num_agents):
            self._update_critic(i, states, actions, rewards, next_states, dones)
            self._update_actor(i, states, actions)

        self.update_target_networks()

    def _update_critic(self, agent_index, states, actions, rewards, next_states, dones):
        state = np.concatenate([states[:, i, :] for i in range(self.num_agents)], axis=1)
        next_state = np.concatenate([next_states[:, i, :] for i in range(self.num_agents)], axis=1)
        action = np.concatenate([actions[:, i, :] for i in range(self.num_agents)], axis=1)

        # # Debugging prints for state and action concatenation
        # print(f"Concatenated State shape: {state.shape}")
        # print(f"Concatenated Next State shape: {next_state.shape}")
        # print(f"Concatenated Action shape: {action.shape}")

        with tf.GradientTape() as tape:
            target_actions = [self.target_actors[i](next_states[:, i, :]) for i in range(self.num_agents)]
            target_actions = tf.concat(target_actions, axis=1)
            
            # # Debugging prints for target actions
            # print(f"Target Actions shape: {target_actions.shape}")

            target_q = self.target_critics[agent_index](next_state, target_actions)
            target_q = rewards[:, agent_index] + self.gamma * target_q * (1.0 - dones[:, agent_index])
            q = self.critics[agent_index](state, action)
            critic_loss = tf.reduce_mean(tf.square(target_q - q))

        critic_gradients = tape.gradient(critic_loss, self.critics[agent_index].trainable_variables)
        self.critic_optimizers[agent_index].apply_gradients(zip(critic_gradients, self.critics[agent_index].trainable_variables))

    def _update_actor(self, agent_index, states, actions):
        state = np.concatenate([states[:, i, :] for i in range(self.num_agents)], axis=1)

        with tf.GradientTape() as tape:
            actions_pred = [self.actors[i](states[:, i, :]) if i == agent_index else actions[:, i, :] for i in range(self.num_agents)]
            actions_pred = tf.concat(actions_pred, axis=1)
            q = self.critics[agent_index](state, actions_pred)
            actor_loss = -tf.reduce_mean(q)

        actor_gradients = tape.gradient(actor_loss, self.actors[agent_index].trainable_variables)
        self.actor_optimizers[agent_index].apply_gradients(zip(actor_gradients, self.actors[agent_index].trainable_variables))
