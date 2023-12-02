import random
import numpy as np
import collections
import torch
import torch.nn.functional as F


class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
    
class Qnet(torch.nn.Module):
    ''' 2 hidden layer Q network '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim[0]*state_dim[1]*state_dim[2], hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, int(hidden_dim/2))
        self.fc3 = torch.nn.Linear(int(hidden_dim/2), action_dim)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))  # use ReLU activation
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)
        
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # Use Adam optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # decay factor
        self.epsilon = epsilon  # epsilon-greedy
        self.target_update = target_update  # target network update freq
        self.count = 0 
        self.device = device

    def take_action(self, state):  # epsilon-greedy policy
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(
                transition_dict['states'],
                dtype=torch.float
            ).to(self.device)
        actions = torch.tensor(
                transition_dict['actions']
            ).view(-1, 1).to(self.device)
        rewards = torch.tensor(
                transition_dict['rewards'],
                dtype=torch.float
            ).view(-1, 1).to(self.device)
        next_states = torch.tensor(
                transition_dict['next_states'],
                dtype=torch.float
            ).to(self.device)
        dones = torch.tensor(
                transition_dict['dones'],
                dtype=torch.float
            ).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        # q_vals for the next states
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = \
            rewards + self.gamma * max_next_q_values * (1 - dones)  # TD target
        dqn_loss = torch.mean(
                F.mse_loss(q_values, q_targets)
            )  # use MSE loss
        self.optimizer.zero_grad() # clean grad explicitly
        dqn_loss.backward()  # update params
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())
        self.count += 1