""" Following Sorscher et al. 2023,
# here we want to generate hexagonal grid cells from path integration over place cell activations
# we implement this with RatInABox

## This notebook is quite long, containing all of the following:

- 1. trajectory generation using ratinabox

- 2. RNN model specification and training using PyTorch

"""
## setup ##

import ratinabox as riab
from ratinabox.Neurons import PlaceCells
import torch

from tqdm import tqdm # keep track of progress

## global variables ##
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# copying over from Sorscher et al.'s repo:
class Options:
    pass
options = Options()

options.device = DEVICE
options.save_dir = '/mnt/fs2/bsorsch/grid_cells/models/'

options.n_steps = 10000      # number of training steps
options.batch_size = 200      # number of trajectories per batch
options.sequence_length = 20  # number of steps in trajectory
options.n_epochs = 100        # number of epochs
options.environment_scale = 1.5 # width and height (m) of square training environment
options.framerate = 1/60      # data rate for both position and place cell firingrate
options.learning_rate = 1e-4  # gradient descent learning rate
options.Ng = 4096             # number of grid cells
options.Np = 512              # number of place cells
options.place_cell_rf = 0.12  # width of place cell center tuning curve (m)
options.pc_type = "diff_of_gaussians" # 'diff of gaussians' or 'gaussian'
options.RNN_type = 'RNN'      # RNN or LSTM
options.activation = 'relu'   # recurrent nonlinearity
options.weight_decay = 1e-4   # strength of weight decay on recurrent weights
#options.periodic = False      # trajectories with periodic boundary conditions
#options.surround_scale = 2    # if DoG, ratio of sigma2^2 to sigma1^2 
#^^ NB: ratinabox has this default ot 1.5

# define the model #

class vel2pc_RNN(torch.nn.Module):
    def __init__(self, options):
        super(vel2pc_RNN, self).__init__()

        self.hidden_size = options.Ng
        self.input_size = 2 # Input will be [velocity_x, velocity_y]
        self.output_size = options.Np

        # Simple RNN layer
        if options.RNN_type == 'RNN':
            self.rnn = torch.nn.RNN(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=1,
                batch_first=True  # (batch, seq, feature)
            )
        elif options.RNN_type == 'LSTM':
            self.rnn = torch.nn.LSTM(input_size = self.input_size,
                                     hidden_size = self.hidden_size)

        if options.activation == 'relu':
            self.activation = torch.nn.functional.relu
        elif options.activation == 'tanh':
            self.activation = torch.nn.functional.tanh
        
        # Fully connected output layer to predict next position as a place-cell rate code.
        self.fc = torch.nn.Linear(self.hidden_size, self.output_size)
        
        # for computing the loss, we set the weight decay
        self.weight_decay = options.weight_decay
        
        # make sure the model and its parameters are on the specified device
        self.device = options.device
        self.to(self.device) 

    def forward(self, input, hidden_state = None):
        '''Takes velocity input as batch_first #(batch,seq,2)'''
        # Initialize hidden state if not provided
        if hidden_state is None:
            h0 = torch.zeros(1, input.size(0), self.hidden_size).to(input.device)
        else:
            h0 = hidden_state
        out, hidden = self.rnn(input, h0)
        out = self.activation(out)
        out = self.fc(out)
        return out, hidden
    
    def compute_loss(self, outputs, targets):
        '''Losses are not usually considered part of a model, 
        but we keep them here as they are crucial for hidden unit representations'''
        # MSE loss for next step prediction
        mse_loss = torch.nn.MSELoss()(outputs, targets)
        # L1 regularization term
        weights = []
        for param in self.parameters():
            weights.append(param.view(-1))
        weights = torch.cat(weights)
        l1_reg = self.weight_decay * torch.norm(weights, p=1)
        # Total loss
        total_loss = mse_loss + l1_reg
        return total_loss, mse_loss, l1_reg


class vel2pc_dataset(torch.utils.data.Dataset):
    '''Creating a dataset that is friendly towards Torch dataloaders for fast training.
    Should be a list of tuples (input, target).
    here input is (seq_length, 2); velocities at t_n
         target is (seq_length, n_place_cells); firingrates at t_n+1
         
    Note: has self.rat and self.place_cells from riab to generate pandas dataframes'''
    
    def __init__(self, options):
        # we generate data using ratinabox
        self.box = riab.Environment(params = {'scale':options.environment_scale})
        self.rat = riab.Agent(self.box, params={"dt":options.framerate})
        self.place_cells = PlaceCells(self.rat, params = {"n":options.Np,
                                                          "description":options.pc_type,
                                                          })
        self.num_samples = options.n_steps
        self.seq_length = options.sequence_length
        self.n_sequences = (self.num_samples-1)//self.seq_length #used to reshape data later

        self.data = [] #should be a list of tuples (input, target)
        self.generate_data()

    def generate_data(self):
        for _ in tqdm(range(self.num_samples)):
            self.rat.update()
            self.place_cells.update()
        #note that we do some reshaping here so velocities at time t_n target firingrates at t_n+1
        velocity_array = torch.tensor(self.rat.history['vel'])[:self.n_sequences*self.seq_length]
        firingrate_array = torch.tensor(self.place_cells.history['firingrate'])[1:self.n_sequences*self.seq_length+1]
        reshaped_velocity = velocity_array.reshape(self.n_sequences,self.seq_length,2)
        reshaped_firingrates = firingrate_array.reshape(self.n_sequences,self.seq_length, firingrate_array.shape[-1])

        self.data = [(reshaped_velocity[i],reshaped_firingrates[i]) for i in range(self.n_sequences)]

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        return self.data[idx]
## functions ##
  
#with our model and training data, we can define a training loop!

def train_model(options = options):
    #set up our rnn
    model = vel2pc_RNN(options)
    
    print('Generating data!')
    dataset = vel2pc_dataset(options)
    train_size = int(0.8*len(dataset))
    val_size = len(dataset)-train_size
    train_data, validation_data = torch.utils.data.random_split(dataset,[train_size, val_size])
    
    #set up optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr = options.learning_rate)
    
    print('Training model!')
    #store all the losses during training
    losses = {'train':{'total':[],'mse':[],'l1':[]},
              'validation':{'total':[],'mse':[],'l1':[]}} 
    for epoch in tqdm(range(options.n_epochs)):
        for data_type, data in {'train':train_data,'validation':validation_data}.items():
            data_loader = torch.utils.data.DataLoader(data,
                                                batch_size = options.batch_size, 
                                                shuffle = True) 
            total_loss = 0
            total_mse = 0
            total_l1 = 0
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(options.device), targets.to(options.device)
                optimizer.zero_grad()
                # Forward pass
                outputs, _ = model(inputs)
                # Calculate loss (MSE between predicted and true next position)
                loss, mse_loss, l1_loss = model.compute_loss(outputs, targets)
                loss.backward() # Backward pass
                optimizer.step() # Update parameters
                total_loss += loss.item()
            losses[data_type]['total'].append(total_loss / len(data_loader)) 
            losses[data_type]['mse'].append(total_mse / len(data_loader))
            losses[data_type]['l1'].append(total_l1 / len(data_loader))

    return model, losses
