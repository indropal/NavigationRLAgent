import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
from collections import namedtuple, deque

from DeepQAgent.QNetwork import QNetwork

# H-parameters & Fixed Setup params
BUFFER_SIZE = int(6*1e3)# size of the memory buffer
PRIORITY_BIAS = 0.0005  # bias added to prioriy to avoid zero priority
SAMPLE_SIZE = 64        # size of sample from the buffer memory

GAMMA = 0.95            # discount factor
TAU = 5e-3              # for soft update of target parameters
LR = 5e-3               # learning rate 
UPDATE_TSTEP = 8        # number of time_steps after which we update the QNetwork


class Agent():
    """
       Definition of Agent to interact with the environment & converge on an optimal strategy
       via Deep Q-Network techniques
    """
    def __init__(self, state_size, action_size, seed, device):
        """
            Initialize the Agent object with parameters t enable the Agent to
            interact with the environment
            
            Args:
                state_size: int : Size of the feature vector representation of the environment state
                action_size: int : Size of the action-space / number of discrete actions possible by Agent
                device: torch.device : Device cpu / gpu where the compute is to be performed 
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.device = device
        self.gamma = GAMMA
        
        # define the QNetworks
        self.qnet_local = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)  # Local netwrok for optimization 
        self.qnet_target = QNetwork(self.state_size, self.action_size, self.seed).to(self.device) # Target network  
        
        # get the optimizer
        self.optimizer = optim.RMSprop( self.qnet_local.parameters(), lr=LR, alpha=0.99, eps=5e-07,
                                        weight_decay=0.05, momentum=0.25, centered=False
                                      )
        
        # Instantiate the Replay Buffer
        self.buffer = ExpBuffer( SAMPLE_SIZE, self.seed, self.device )
        
        # initialize the timestep
        self.t_step = 0
        
    def act(self, state, epsilon = 1.0):
        """
            Obtain the Agent's Action for a given State which is according to 
            the Learnt policy
            
            Args: 
                state : List[float] -> this is the feature vector used to define the State
                epsilon : float -> epsilon value to determin if greedy approach or not
                
            Returns:
                Tuple of : 
                        > Discrete Action taken by the Agent according to the policy : int
                        > Action-Value of the chosen action by the Agent : float
        """
        state = torch.from_numpy( np.array(state) ).float().unsqueeze(0).to(self.device)
        
        # set to evaluation mode / no gradients computed
        self.qnet_local.eval()  
        
        action_values = torch.from_numpy(np.array([])).float()
        
        with torch.no_grad():
            # get the respective action-values for the passed state
            action_values = self.qnet_local(state)
            
        # set the network back in train-mode i.e. allowing compute of gradients
        self.qnet_local.train()
        
        agent_action_step = 0.0
        
        if random.random() > epsilon:
            # take the greedy-approach -> choose action for one with highest value
            agent_action_step = np.argmax(action_values.cpu().data.numpy())
        else:
            # Explore different values
            agent_action_step = np.random.choice(np.arange(self.action_size))
            
        return ( agent_action_step.astype('int32'), action_values.cpu().data.numpy()[0][agent_action_step] )

    def step(self, estimate, state, action, reward, next_state, done):
        """
            Performs Adding the experience tuple obtained from the environment to the Replay Buffer
            & initiating QNetwork train and update of target Q Network
            
            Args:
                estimate: float -> observed Q-value for state-action pair
                
                state: numpy.array() / torch.tensor -> tensor/array which represents the current state
                action: int -> action taken by the Agent
                reward: float -> reward value obtained by the Agent
                next_state: numpy.array() / torch.tensor -> tensor/array representing the next state
                done: bool -> boolean value represnting if the Episode has completed / terminal state has been reached
        """
        
        next_state = torch.from_numpy( np.array(next_state) ).float().unsqueeze(0).to(self.device)
        target = torch.tensor([]).float()
        
        # Compute the Q-target from the 'reward' and 'next_state' values
        self.qnet_target.eval()  # set the Target Network in Evaluation Mode
        
        with torch.no_grad():
            target = self.qnet_target(next_state) # return a tensor with action values for all discrete actions
            target = target.cpu().data.numpy().max() # Greedy --> Taking max value from the Target Network
  
        self.qnet_target.train() # set the Target network back into Train Mode
        
        # convert 'next_state''back to np.ndarray
        next_state = next_state.cpu().data.numpy(); # print(next_state.shape)
        
        # Include the experience tuple into the Replay-Buffer
        self.buffer.add_exp( target, estimate, state, action, reward, next_state, done )
        
        self.t_step = (self.t_step + 1) % UPDATE_TSTEP # increment the timestep counter until 'UPDATE_TSTEP' elapsed
        
        # print('In Agent.step() -> Right outside of learn-trigger check | Buffer Size: {} | t: {}'.format(len(self.buffer),self.t_step) )
        
        if (self.t_step == 0) and (SAMPLE_SIZE <= len(self.buffer) ):
            # if 'UPDATE_TSTEP' number of timesteps have elapsed - Update the Network
            # if enough samples are present in the Replay Buffer -> initiate QNetwork Learning
            
            # print('Triggering Learn | Buffer Size: {} | t: {}'.format(len(self.buffer),self.t_step))
            
            experiences = self.buffer.sample()  # take a sample from the Buffer
            self.learn(experiences, self.gamma)
    
    
    def learn( self, experiences, GAMMA, hparam_b = 0.85):
        """
            Initiate Learning of the QNetwork
            Args:
                experience : tuple(torch.tensor) -> tuple of torch.tensors containing (state, action, reward, next_state, done)
                GAMMA : float -> constant value defining the discount rate
        """
        exp_states, exp_actions, exp_rewards, exp_next_states, exp_dones, exp_p = experiences # each arranged in np.vstack fashion
        
        # Get max predicted Q-values (for next states) from target model
        Q_target = self.qnet_target(exp_next_states).detach().max(1)[0].unsqueeze(1) # corresponds to Q(next_state, max_action)
        
        # Target for the current state ->
        Q_target = exp_rewards + self.gamma*Q_target*(1 - exp_dones) # we donot want to consider terminal state ie. done = True
        
        # Get expected Q values from local Q-network -> 'gather' is used to isolate Q-values for corresponding actions taken
        Q_expected = self.qnet_local(exp_states).gather(1, exp_actions)
        
        # Compute the Loss along with Bias adjustment
        loss = (Q_expected - Q_target)**2
        # loss = torch.nn.functional.mse_loss(Q_expected, Q_target)
        
        # Priority Bias adjustment
        bias = ((1/len(self.buffer))*(1/exp_p))**hparam_b
        
        # MSE-LOSS with Priority-Bias adjustment
        loss = torch.mean(loss*bias); # print(loss)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnet_local, self.qnet_target, TAU) 
    
    def soft_update(self, local_qnet, target_qnet, tau):
        
        for target_param, local_param in zip(target_qnet.parameters(), local_qnet.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
        
# Define the Replay Buffer
class ExpBuffer():
    """
        Replay Buffer used to store state-transition tuples in format of : 'state', 'action', 'reward', 'next_state', 'done', 'priority'
        These tuples are used as experience in order to break correlation between 'action'<->'next_state' also, 'state'<->'next_state'
        by sampling them at random from the Buffer.
        
        This implementation of Replay-Buffer implements a 'Prioritized Experience Replay' strategy where the experience-tuples are sampled
        based on the priority.
    """
    
    def __init__(self, sample_size, seed, device):
        """
            Initialise the Buffer object
            Args : 
                sample_size : int -> size of the sample to be returned from the Buffer
                seed : int -> seed value for bemchmarking in random processes
                device : torch.device -> device allotment for torch processing
        """
        self.experience = namedtuple( 'experience', ['state', 'action', 'reward', 'next_state', 'done', 'priority'])
        self.delta = deque(maxlen = BUFFER_SIZE)
        self.memory = deque(maxlen = BUFFER_SIZE)
        self.priorities = []
        self.sample_size = sample_size
        self.seed = seed
        self.device = device
        
        self.release_mem_size = self.sample_size//2
        
    def __len__(self):
        """
            return the length of the Replay Buffer - the number of experience tuples inside 
        """
        return len(self.memory)
    
    def sample(self):
        """
            Return a sample of experience tuples taken from the buffer.
        """
        priority = [ e.priority for e in self.memory if e is not None ]
        experiences = random.choices( self.memory, weights = priority, k = self.sample_size )
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        priorities=torch.from_numpy(np.vstack([e.priority for e in experiences if e is not None])).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones, priorities)
        
    def add_exp(self, target, estimate, state, action, reward, next_state, done, hparam_a = 0.95):
        """
            Include experience tuples into the Replay Buffer And also compute their associated Priority Values - 
            along with revising all the priority values in the Buffer
            
            Args:
                target: float -> target Q-value for state, action pair
                estimate: float -> observed Q-value for state-action pair
                state: List[float] -> List of features which represents the current state
                action: int -> action taken by the Agent
                reward: float -> reward value obtained by the Agent
                next_state: List[float] -> List of features representing the next state
                done: bool -> boolean value represnting if the Episode has completed / terminal state has been reached   
        """
        
        # find the delta for the new tuple
        new_delta = (abs(estimate - target) + PRIORITY_BIAS) # bias added to prevent zero value
        
        # If there is a memory overflow - only retain the ones with highest 'priority'
        if len(self.delta) == BUFFER_SIZE:
            """
            # pop the one with least value of delta -> find the index with least value
            idx, min_d = 0, float('inf')
            
            # find element-index of least value of delta
            for i, d in enumerate(self.delta):
                if d < min_d:
                    idx = i
                    min_d = d
            
            del self.delta[idx]  # delete the specific delta value
            del self.memory[idx] # also delete corresponding experience tuple
            """
            # Delete 'self.release_mem_size' equivalent items from memory for the smallest values of 'priority'
            self.memory = deque( sorted(self.memory, key = lambda m: m.priority)[-self.release_mem_size:], maxlen = BUFFER_SIZE )
            self.delta = deque( sorted(self.delta)[-self.release_mem_size:], maxlen = BUFFER_SIZE )
            
        # include 'delta' into the current store of delta values
        self.delta.append(new_delta)
        
        # compute the Priority values
        self.priorities = [p**hparam_a for p in self.delta]
        sum_denom = sum(self.priorities);# print(self.priorities, sum_denom); print('---------------------------------')
        
        # recompute the priority values -> NORMALIZE THEM
        self.priorities = [ self.priorities[i] / sum_denom for i in range(len(self.delta))]
        # print(self.priorities, sum_denom); print('---------------------------------')
        
        # restore the priorities in each experience tuple in Buffer Memory
        for i in range(len(self.memory)):
            # self.priorities[i] = self.priorities[i].detach().cpu()
            self.priorities[i] = np.round( self.priorities[i], decimals = 10 ); #print(self.priorities[i])
            self.memory[i] = self.experience( self.memory[i].state, self.memory[i].action, self.memory[i].reward,
                                              self.memory[i].next_state, self.memory[i].done, self.priorities[i]
                                            )
        # append the new tuple into memory
        # self.priorities[-1] = self.priorities[-1].detach().cpu();
        self.priorities[-1] = np.round( self.priorities[-1], decimals = 10 ); # print(self.priorities[-1])
        self.memory.append( self.experience( state, action, reward, next_state, done, self.priorities[-1] )
                          )   
        
    def get_priorities(self):
        """
            Return the List of priorities
        """
        return self.priorities