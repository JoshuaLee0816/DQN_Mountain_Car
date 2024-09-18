import gymnasium as gym
import torch
import torch.nn as nn
import random
from collections import deque

# Step 1: Build environment
env = gym.make("MountainCar-v0")

# Step 2: Build Q-Network, input = state, output = q value
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size): #Define the basic network structure
        super(QNetwork, self).__init__() # Calls the parent class's __init__ method
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, state):               #Forward pass, neural network basic transmit
        x = torch.relu(self.fc1(state))     #Relu = max(0, x), transmit into first layer, by using matrix manipulation, then Relu is the second time transmition.
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:                         #In order to break the correlation between continuous data, improve ability of generalization
    def __init__(self, buffer_size):
        self.buffer = deque( maxlen = buffer_size )
    
    def add(self, state, action, reward, next_state, done):
        experience = (torch.tensor(state, dtype = torch.float32),
                      torch.tensor(action, dtype = torch.long),
                      torch.tensor(reward, dtype = torch.float32),
                      torch.tensor(next_state, dtype = torch.float32),
                      torch.tensor(done, dtype = torch.float32))
        
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)   #zipped the corresponding data together from different lists

        # Convert tuple into Tensor // Tensor is very important in pyTorch RL Training
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)  
        rewards = torch.stack(rewards).unsqueeze(1)
        next_states = torch.stack(next_states)
        dones = torch.stack(dones).unsqueeze(1)

        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

#Step 3 , Hyperparameters Setting
Buffer_Size, Batch_Size, Gamma, Learning_Rate = 10000, 64, 0.99, 0.0001
Epsilon, Min_Epsilon, Epsilon_Decay = 1, 0.01, 0.995
Target_Update, Num_Episodes, Max_Steps = 10, 1000, 200

#Step 4, Initialize Network and BufferReplay
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

policy_Net = QNetwork(state_size, action_size)  #policy_network updates every episode, while target_network doesn't
target_Net = QNetwork(state_size, action_size)  #Two networks for the sake of maintaining training stability
target_Net.load_state_dict(policy_Net.state_dict())  #Initial same dict first
target_Net.eval()                               #Set as evaluation mode[The above are all initializations]

replay_Buffer = ReplayBuffer(Buffer_Size)

#Step 5 , Define Optimizer and Criterion
optimizer = torch.optim.Adam(policy_Net.parameters(), lr = Learning_Rate)
criterion = nn.MSELoss()                        #Consider other loss function that may be used

#Step 6 , Define Epsilon_Greedy Strategy
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():                   #Forward Passing doesn't need to calculate gradient descent ,to speed up
            q_values= policy_Net(state)
            return q_values.argmax().item()
        
#Step 7 , Define a function to operate one time training process, include sampling from Replay Buffer, Q_value Calculation, Loss Calculation, and Update.
def optimize_model():
    if len(replay_Buffer) < Batch_Size:
        return                                  #If the len of buffer experience is not enough for a batch, then skip
    
    #Sampling from Replay Buffer
    states, actions, rewards, next_states, dones = replay_Buffer.sample(Batch_Size)

    #Current Q_value Calculation
    current_q_values = policy_Net(states).gather(1, actions)

    #Objective Q_value Calculation
    with torch.no_grad():
        max_next_q_values = target_Net(next_states).max(1)[0]
        target_q_values = rewards + (Gamma * max_next_q_values * (1-dones))

    #Adjust Objective Q_values to match current q values
    target_q_values = target_q_values.unsqueeze(1)

    #Loss Calculation
    loss = criterion(current_q_values, target_q_values)

    #Update
    optimizer.zero_grad()   #Avoid accumulation of gradient
    loss.backward()         #Calculate gradient of loss to network parameters
    optimizer.step()        #Update network parameters

#Step 8 , main training Loop

for episode in range(1, Num_Episodes + 1):
    state, _ = env.reset()
    state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)     #Convert into Tensor
    total_reward = 0

    for t in range(Max_Steps):
        action = select_action(state, Epsilon)

        #Execute action and get next state and reward
        next_state, reward, done,_,_ = env.step(action)
        total_reward += reward

        #Convert to Tensor
        next_state = torch.tensor(next_state, dtype = torch.float32).unsqueeze(0)
        reward = torch.tensor([reward], dtype = torch.float32)
        done = torch.tensor([done], dtype = torch.float32)

        #Store experience into ReplayBuffer
        replay_Buffer.add(state.squeeze(0).numpy(), action, reward.item(), next_state.squeeze(0).numpy(), done.item())

        #Update State
        state = next_state

        #Optimize
        optimize_model()

        if done:
            break

    #Update Epsilon
    Epsilon = max(Min_Epsilon, Epsilon_Decay * Epsilon)

    #Update target network in a certain period
    if episode % Target_Update == 0:
        target_Net.load_state_dict(policy_Net.state_dict())

    if episode & 10 == 0:
        print(f"Episode {episode}/{Num_Episodes}, Total Reward: {total_reward}, Epsilon: {Epsilon:.4f}")

print("Training Accomplish!!!")

#Step 9 , Save the model
torch.save(policy_Net.state_dict(), "DQN_MountainCar.pth")
print("Model Saved")