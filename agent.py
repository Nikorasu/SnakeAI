#!/usr/bin/env python3
from snake import do, print_state
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from time import sleep

# Define the neural network
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Define the replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def push(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

        if len(self.states) > self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)

    def sample(self, batch_size):
        indices = random.sample(range(len(self.states)), batch_size)
        states = [self.states[i] for i in indices]
        actions = [self.actions[i] for i in indices]
        rewards = [self.rewards[i] for i in indices]
        next_states = [self.next_states[i] for i in indices]
        dones = [self.dones[i] for i in indices]
        return states, actions, rewards, next_states, dones

# Define the DQN agent
class DQNAgent:
    def __init__(self, input_size, output_size, buffer_size=10000, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, target_update_frequency=1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN(input_size, output_size).to(self.device)
        self.target_dqn = DQN(input_size, output_size).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters())
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency
        self.update_counter = 0

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q_values = self.dqn(state)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(q_values.size(-1))
        else:
            action = q_values.max(1)[1].item()
        return action

    def update(self):
        if len(self.buffer.states) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.stack([torch.from_numpy(state).to(self.device) for state in states]).float()
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.stack([torch.from_numpy(next_state).to(self.device) for next_state in next_states]).float()
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        q_values = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_dqn(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

# Initialize the agent and environment
input_size = 64  # 8x8 board size
output_size = 3  # 3 actions (left, forward, right)
agent = DQNAgent(input_size, output_size, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999, target_update_frequency=1000)

# Training loop
episodes = 10000

for episode in range(episodes):
    # Initialize the game
    board_size = 8
    snake = torch.zeros((board_size, board_size), dtype=torch.int)
    snake[0, :4] = torch.tensor([1, 2, 3, -1]) # starting this long, so learns not to loop sooner
    score = do(snake, 1)  # needed so snake becomes 4 long and first random food spawns

    # Print initial state
    print_state(snake)
    print(f" H:{snake.max().item():>2}  R:{score:>2}  Ep:{episode}")

    state = snake.flatten().numpy()
    done = False
    episode_reward = 0
    last_food = 0

    while not done:
        action = agent.get_action(state)
        score = do(snake, action)
        next_state = snake.flatten().numpy()
        reward = score if score != -1 else -10 #snake.max().item()-4
        #reward = score if score != 0 else -10
        # if reward is 10, add snake length to score
        if reward == 10:
            #reward += (snake.max().item() - 4) *2
            last_food = 0  # Reset the step count if food was eaten
        else:
            last_food += 1  # Increment the step count
        if last_food > 32 or score == -1:
            if last_food > 32:
                reward = -20
            done = True  # End the episode if the snake has not eaten food for too long

        agent.buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        
        # Print the updated state and score
        print_state(snake)
        print(f" H:{snake.max().item():>2}  R:{reward:^3}  TO:{32-last_food:>2}  Ep:{episode}  eR:{episode_reward}")
        sleep(.02)

        agent.update()
