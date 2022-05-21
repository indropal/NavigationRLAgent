import torch
import torch.optim as optim

class QNetwork(torch.nn.Module):
    """
        Definition of the Q-Network architecture which is used to map
        a 'state' to corresponding expected 'action'-values / Q-values 
    """
    def __init__(self, state_size, action_size, seed,
                 l1_n_units = 128, l2_n_units = 1024, l3_n_units = 1024,
                 l4_n_units = 64, l5_n_units = 32, l6_n_units = 16
                ):
        """
            Initialization of the Q-Network model.
            
            Args:
                l1_n_units : Number of units in L1 layer
                l2_n_units : Number of units in L2 layer
                l3_n_units : Number of units in L3 layer
                l4_n_units : Number of units in L4 layer
                l5_n_units : Number of units in L5 layer
                
                state_size : size of tensor / vector representing a state
                action_size : number of possible discrete action available in the Environment
                seed : Seed value for random initializations

                [ver 5]
                 l1_n_units = 256, l2_n_units = 1024, l3_n_units = 1024,
                 l4_n_units = 256                   
                -------------------------------------------------------
                [ver 4]
                 l1_n_units = 128, l2_n_units = 64, l3_n_units = 256,
                 l4_n_units = 128, l5_n_units = 64, l6_n_units = 16                    
                -------------------------------------------------------
                [ver 3]
                 l1_n_units = 128, l2_n_units = 256, l3_n_units = 64,
                 l4_n_units = 16                   
                -------------------------------------------------------
                [ver 2]
                 l1_n_units = 64, l2_n_units = 128, l3_n_units = 64,
                 l4_n_units = 16                
                -------------------------------------------------------
                 [ver 1]
                 l1_n_units = 64, l2_n_units = 128, l3_n_units = 128,
                 l4_n_units = 64
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        
        self.action_size = action_size
        self.state_size = state_size
        
        # instantiate the Netwrk's layers
        self.fc1 = torch.nn.Linear(self.state_size, l1_n_units)
        self.fc2 = torch.nn.Linear(l1_n_units, l2_n_units)
        self.fc3 = torch.nn.Linear(l2_n_units, l3_n_units) # self.fc3 = torch.nn.Linear(l2_n_units, action_size)
        self.fc4 = torch.nn.Linear(l3_n_units, l4_n_units)
        self.fc5 = torch.nn.Linear(l4_n_units, action_size)
        
        # self.fc4 = torch.nn.Linear(l3_n_units, l4_n_units)
        # self.fc5 = torch.nn.Linear(l4_n_units, action_size) # Old Version
        
        #self.fc5 = torch.nn.Linear(l4_n_units, l5_n_units)
        #self.fc6 = torch.nn.Linear(l5_n_units, action_size)
        """
        self.fc6 = torch.nn.Linear(l5_n_units, l6_n_units)
        self.fc7 = torch.nn.Linear(l6_n_units, action_size)
        """
        
        
    def forward(self, state):
        """
            Feed-forward Network architecture
            
            Args:
                state: torch.tensor -> input to the Network
                
            Return:
                torch.tensor -> which gives Network's feed-forward value for input tensor
        """
        l1 = self.fc1(state)
        l1 = torch.nn.functional.relu(l1) # torch.nn.functional.leaky_relu(l1, negative_slope = 0.02)
        
        l2 = self.fc2(l1)
        l2 = torch.nn.functional.relu(l2) # torch.nn.functional.leaky_relu(l2, negative_slope = 0.02)
        
        l3 = self.fc3(l2)
        l3 = torch.nn.functional.relu(l3)
        
        l4 = self.fc4(l3)
        l4 = torch.nn.functional.relu(l4)
        
        l5 = self.fc5(l4)
        # l5 = torch.nn.functional.relu(l5)
        
        return l5
        
        """
        l5 = self.fc5(l4)
        l5 = torch.nn.functional.relu(l5)
        
        #return self.fc5(l4)
        return self.fc6(l5)
        """
        """
        l5 = self.fc5(l4)
        l5 = torch.nn.functional.relu(l5)
        
        l6 = self.fc6(l5)
        l6 = torch.nn.functional.relu(l6)
        
        return self.fc7(l6)
        """
        
        
#     def backward(self, loss_criterion, target, estimate, optimizer):
#         """
#             Back-Propagation for calculating gradients of all
#             the Network parameters and updating them via the optimizer
            
#             Args:
#                 loss_criterion : torch.nn.functional -> loss function
#                 target : torch.tensor -> target value
#                 estimate : torch.tensor -> estimated value of target
#                 optimizer : torch.optim -> Optimizer function
               
#         """
        
#         # compute the Loss value
#         loss = loss_criterion(target, estimate)
        
#         # Clear out the computed gradients in the optimizer
#         optim.zero_grad()
        
#         # Initiate backward Propagation from the Loss
#         loss.backward()
        
#         # initiate Update-Step of the Optimizer by computing the gradients
#         optim.step()
       