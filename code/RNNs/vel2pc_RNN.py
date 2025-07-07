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
import numpy as np # for numerical operations

import json
from tqdm import tqdm # keep track of progress
from pathlib import Path # for saving models and generated data
import hashlib # generate ID's for saving models
import pandas as pd #saving losses as readable format

#local imports
from GridModels.code.functions import position_decoders as decoding

## global variables ##
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# copying over from Sorscher et al.'s repo:
class Options:
    pass
options = Options()

options.device = DEVICE
options.save_dir = '../code/GridModels/data/sorscher_models/'

options.n_steps = 100000      # number of steps in full trajectory
options.batch_size = 200      # number of trajectories per batch
options.sequence_length = 20  # number of steps in trajectory sub-sequence (for training)
options.n_epochs = 100       # number of epochs during training
options.train_ratio = 0.8    # proportion of data used for training (rest is validation)
options.environment_scale = 1.5 # width and height (m) of square training environment
options.framerate = 1/60      # data rate for both position and place cell firingrate
options.learning_rate = 1e-4  # gradient descent learning rate
options.Ng = 4096             # number of grid cells
options.Np = 512              # number of place cells
options.place_cell_rf = 0.15  # width of place cell center tuning curve (m)
options.pc_type = "diff_of_gaussians" # 'diff of gaussians' or 'gaussian'
options.RNN_type = 'RNN'      # RNN or LSTM
options.activation = 'relu'   # recurrent nonlinearity
options.weight_decay = 1e-6   # strength of weight decay on recurrent weights
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

        # We need to both reset hidden state (PC->GC) and decode (GC -> PC)
        self.encoder = torch.nn.Linear(self.output_size, self.hidden_size)
        self.decoder = torch.nn.Linear(self.hidden_size, self.output_size)
        
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
        out = self.decoder(out)
        return out, hidden
    
    def compute_loss(self, outputs, targets):
        ''' '''
        # MSE loss for next step prediction
        mse_loss = torch.nn.MSELoss()(outputs, targets)
        # L1 regularization term
        weights = []
        for param in self.parameters():
            weights.append(param.view(-1))
        weights = torch.cat(weights)
        l1_reg = self.weight_decay * (self.rnn.weight_hh_l0**2).sum()
        # Total loss
        total_loss = mse_loss + l1_reg
        return total_loss, mse_loss, l1_reg


class vel2pc_dataset(torch.utils.data.Dataset):
    '''Creating a dataset that is friendly towards Torch dataloaders for fast training.
    Should be a list of tuples (input, target).
    here input is (seq_length, 2); velocities at t_n
         target is (seq_length, n_place_cells); firingrates at t_n+1
'''
    
    def __init__(self, options, navigation_rates_df = None):
        # we generate data using ratinabox
        self.num_samples = options.n_steps
        self.seq_length = options.sequence_length
        self.n_sequences = (self.num_samples-1)//self.seq_length #used to reshape data later
        self.navigation_rates_df = navigation_rates_df
        self.data = [] #should be a list of tuples (input, target)
        self.generate_data()

    def generate_data(self):
        if self.navigation_rates_df is None:
            self.navigation_rates_df = generate_navigation_rates_df(options = options)
        #note that we do some reshaping here so velocities at time t_n target firingrates at t_n+1
        n_sequences = (self.num_samples-1)//self.seq_length #used to reshape data
        velocity_array = torch.tensor(self.navigation_rates_df.velocity.values, dtype = torch.float32)[:n_sequences*self.seq_length]
        firingrate_array = torch.tensor(self.navigation_rates_df.firingrate.values,dtype = torch.float32)[1:n_sequences*self.seq_length+1]
        reshaped_velocity = velocity_array.reshape(self.n_sequences,self.seq_length,2)
        reshaped_firingrates = firingrate_array.reshape(self.n_sequences,self.seq_length, firingrate_array.shape[-1])
        #we take the firing rate of the position at t_n to set the hidden state of the RNN
        init_firingrates =  torch.tensor(self.navigation_rates_df.firingrate.values,dtype = torch.float32)[:n_sequences*self.seq_length][::self.seq_length]
        self.data = [((reshaped_velocity[i],init_firingrates[i]),reshaped_firingrates[i]) for i in range(self.n_sequences)]
    
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        return self.data[idx]
    
## functions ##
  
#top level function (this is what you might use to instantiate a model)#
def get_model_and_data(options):
    
    if (Path(options.save_dir)/generate_model_id(str(options.__dict__))).exists():
        print('Loading existing model and data!')
        model, dataset, losses = load_model(options)
    else:
        model, dataset, losses = train_model(options)
            
    return model, dataset, losses
  
  
# data generation #
def generate_navigation_rates_df(options, true_trajectory = None):
    '''Using ratinabox to generate a dataframe with navigation information and place cell firing rates.'''
    # set up rat and box
    box = riab.Environment(params = {'scale':options.environment_scale})
    rat = riab.Agent(box, params={"dt":options.framerate})
    place_cells = PlaceCells(rat, params = {"n":options.Np,
                                            "description":options.pc_type,})
    options.n_steps = options.n_steps if true_trajectory is None else len(true_trajectory)
    for i in tqdm(range(options.n_steps)):
        rat.update()
        if true_trajectory is not None:
            # if we have a real trajectory, we use it to set the rat's position
            rat.history['pos'][-1] = true_trajectory['centroid_position'].values[i]
            rat.history['vel'][-1] = true_trajectory['velocity'].values[i]
            rat.history['head_direction'][-1] = [true_trajectory['head_direction'].x.values[i], true_trajectory['head_direction'].y.values[i]]
        place_cells.update()
        
    # now save in a format easy to read and store, consistent with our real data.
    nav_rates_dict = {}
    nav_rates_dict['time'] = rat.history['t']
    nav_rates_dict['centroid_position.x'] = [p[0] for p in rat.history['pos']]
    nav_rates_dict['centroid_position.y'] = [p[1] for p in rat.history['pos']]
    nav_rates_dict['velocity.x'] = [v[0] for v in rat.history['vel']]
    nav_rates_dict['velocity.y'] = [v[1] for v in rat.history['vel']]
    nav_rates_dict['head_direction.x'] = [v[0] for v in rat.history['head_direction']]
    nav_rates_dict['head_direction.y'] = [v[1] for v in rat.history['head_direction']]
    nav_rates_dict['head_direction.degrees'] = np.rad2deg([np.arctan2(v[1],v[0]) for v in rat.history['head_direction']])
    for place_cell in range(options.Np):
        nav_rates_dict[f'firingrate.place_cell_{place_cell}'] = [x[place_cell] for x in place_cells.history['firingrate']]
    navigation_rates_df = pd.DataFrame(nav_rates_dict)
    #make it multicolumn index !
    navigation_rates_df.columns = pd.MultiIndex.from_tuples([
        tuple(col.split('.')) if '.' in col else (col, '') 
        for col in navigation_rates_df.columns])

    return navigation_rates_df 

# model training, saving, and loading #
def train_model(options, save = True):
    #set up our rnn
    model = vel2pc_RNN(options)
    
    print('Generating data!')
    dataset = vel2pc_dataset(options)
    train_size = int(options.train_ratio*len(dataset))
    val_size = len(dataset)-train_size
    train_data, validation_data = torch.utils.data.random_split(dataset,[train_size, val_size])
    
    #set up optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr = options.learning_rate)
    
    print('Training model!')
    #store all the losses during training
    losses = {'train':{'total':[],'mse':[],'l1':[]},
              'validation':{'total':[],'mse':[],'l1':[], 'decoding_error':[]}} 
    for epoch in tqdm(range(options.n_epochs)):
        for data_type, data in {'train':train_data,'validation':validation_data}.items():
            data_loader = torch.utils.data.DataLoader(data,
                                                batch_size = options.batch_size, 
                                                shuffle = True) 
            #count losses for each epoch
            total_loss = 0
            total_mse_loss = 0
            total_l1_loss = 0
            with torch.no_grad() if data_type == 'validation' else torch.enable_grad():
                for batch_idx, (inputs, targets) in enumerate(data_loader):
                    #Prepare inputs and target
                    velocity_inputs, init_firingrates = inputs
                    velocity_inputs, targets = velocity_inputs.to(options.device), targets.to(options.device)
                    init_firingrates = init_firingrates.to(options.device)
                    #reset optimiser
                    optimizer.zero_grad()
                    # Forward pass
                    init_hidden = model.encoder(init_firingrates)
                    outputs, _ = model(velocity_inputs,init_hidden.unsqueeze(0))
                    # Calculate loss (MSE between predicted and true next position)
                    loss, mse_loss, l1_loss = model.compute_loss(outputs, targets)
                    if data_type == 'train':  
                        loss.backward() # Backward pass
                        optimizer.step() # Update parameters
                    total_loss += loss.item()
                    total_mse_loss += mse_loss.item()
                    total_l1_loss += l1_loss.item()
            losses[data_type]['total'].append(total_loss /options.batch_size)
            losses[data_type]['mse'].append(total_mse_loss / options.batch_size)
            losses[data_type]['l1'].append(total_l1_loss/ options.batch_size)
            # evaluate position decoding each epoch
            if data_type == 'validation':
                decoding_errors_cm = evaluate_position_decoding(model, dataset)
                losses[data_type]['decoding_error'].append(np.mean(decoding_errors_cm))
                print(f"Epoch {epoch+1}/{options.n_epochs}, decoding error: {np.mean(decoding_errors_cm):.3f} cm")
    if save:
        print('Saving model and data!')
        save_model_run(options, model, dataset, losses)
        
    return model, dataset, losses

def save_model_run(options, model, dataset, losses):
    # generate a unique model ID based on the options
    model_id = generate_model_id(str(options.__dict__), length=16)
    print(f"Saving model and data with ID: {model_id}")
    files_dir = Path(options.save_dir)/model_id
    Path(files_dir).mkdir(parents=True, exist_ok=True)  # Create directory if not already existing
    #save model
    torch.save(model.state_dict(), files_dir/f"vel2pc_{model_id}.pt")
    #save dataset
    torch.save(dataset, files_dir/f'dataset_{model_id}.pt')
    options_dict = options.__dict__.copy()  # Copy options to avoid modifying the original
    options_dict = {k:str(v) for k,v in options_dict.items()}  # Convert all values to strings
    with open(files_dir/'options.json','w') as f:
        json.dump(options_dict, f, indent=4)
    
    losses_dict = {f'{k}.{key}':v[key] for k, v in losses.items() for key in v.keys()}
    losses_df = pd.DataFrame(losses_dict)
    losses_df.to_csv(files_dir/'losses.htsv', sep = '\t', index=False)
    return 

def load_model(options):
    """Load a model from the specified directory.
    Parameters:
        options (Options): Options object containing the save directory and model ID.
    Returns:
        model (vel2pc_RNN): The loaded model.
    """
    model_id = generate_model_id(str(options.__dict__), length=16)
    files_dir = Path(options.save_dir)/model_id
    model = vel2pc_RNN(options)
    model.load_state_dict(torch.load(files_dir/f"vel2pc_{model_id}.pt", 
                                     weights_only=True))
    losses = pd.read_csv(files_dir/'losses.htsv', sep = '\t')
    losses.columns = pd.MultiIndex.from_tuples(
        [tuple(col.split('.')) if '.' in col else (col,"") for col in losses.columns ]
    )
    try:
        dataset = torch.load(files_dir/f'dataset_{model_id}.pt')
    except Exception as e:
        print(f'failed to load dataset: {e}')
        dataset = None
    return model, dataset, losses 

def generate_model_id(description:str, length=16)->str:
    """Generate a unique model ID from a description string.
    Parameters:
        description (str): Descriptive string of model parameters, usually str(options.__dict__).
        length (int): Length of the returned ID (8 or 16 recommended)
    Returns:
        str: Truncated SHA-256 hash with specified length 
    """
    hash_obj = hashlib.sha256(description.encode('utf-8'))
    return hash_obj.hexdigest()[:length]

# model evaluation #

def generate_hidden_activity(model, options):
    '''Generates new trajectory and passes through model to establish activations of hidden units.
    
    Returns 
    -------
    nav_rates_df: pd.DataFrame()
        centroid_position, velocity, and head_direction, followed by firingrate of place cells and hidden_units.'''
    print('generating trajectory')
    pc_nav_rates_df = generate_navigation_rates_df(options)
    hidden_rates = []
    predicted_pc_frs = []
    print('generating hidden activity')
    for i in tqdm(range(len(pc_nav_rates_df))):
        current_step = pc_nav_rates_df.iloc[i]
        if i % options.sequence_length == 0: # reset hidden rates every 
            init_pc_activity = torch.tensor(current_step.firingrate.values).to(dtype = torch.float32,
                                                                            device = options.device)
            with torch.no_grad():
                last_hidden = model.encoder(init_pc_activity).unsqueeze(0)
            if i == 0:
                hidden_rates.append(last_hidden.cpu().numpy().squeeze())
                predicted_pc_frs.append(np.ones((options.Np))*np.nan) #no prediction, so just all NaN's
                continue
        velocity_input = torch.tensor(current_step.velocity.values).to(dtype = torch.float32, 
                                                                    device = options.device)
        with torch.no_grad():    
            predicted_pc_fr, last_hidden = model(velocity_input.unsqueeze(0), last_hidden)
        predicted_pc_frs.append(predicted_pc_fr.cpu().numpy().squeeze())
        hidden_rates.append(last_hidden.cpu().numpy().squeeze())
    #create dataframes
    hidden_activity_df = pd.DataFrame(np.array(hidden_rates), 
                                      columns= [f'firingrate.hidden_unit_{x}' for x in range(options.Ng)])
    hidden_activity_df.columns = pd.MultiIndex.from_tuples(
        [(col.split('.')) if '.' in col else
         (col,"") for col in hidden_activity_df.columns]
    )
    
    predicted_pc_df = pd.DataFrame(np.array(predicted_pc_frs), 
                                      columns= [f'firingrate.predicted_pc_{x}' for x in range(options.Np)])
    predicted_pc_df.columns = pd.MultiIndex.from_tuples(
        [(col.split('.')) if '.' in col else
         (col,"") for col in predicted_pc_df.columns]
    )
    
    return pc_nav_rates_df, hidden_activity_df, predicted_pc_df
    
def evaluate_position_decoding(model, dataset, decoder_type = 'GP'):
    """Train a decoder on actual place cell firing rates,
        then test on model predicted place cell firing rates in test data. 
        (this is pseudo-double-cross-validation)
        
    Parameters:
    -----------
    
    Returns:
    --------
    position_decoding_errors: np.array()
        decoding errors in cm for each position in the test set.
    """
    split_idx = int(options.n_steps*options.train_ratio) #split data into train and test like in model training
    training_data = dataset.navigation_rates_df
    test_data = dataset.navigation_rates_df[split_idx:]
    gp_model = decoding.train_decoder(training_data[:split_idx],subsample_n=20, 
                                      type = decoder_type)
    #now we want to decode on held out test data:
    predicted_pc_frs= []
    with torch.no_grad():
        #We need to reset model with hippocampal inputs every n_seq steps, as in training
        ordered_dataloader = torch.utils.data.DataLoader(dataset,
                                                         batch_size = options.batch_size, 
                                                         shuffle = False)
        n_validation_batches = 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(ordered_dataloader)):
            if batch_idx*options.batch_size*options.sequence_length < split_idx:
                continue
            if targets.shape[0] < options.batch_size:
                continue
            velocity_inputs, init_firingrates = inputs
            velocity_inputs = velocity_inputs.to(DEVICE, dtype=torch.float32)
            init_firingrates = init_firingrates.to(DEVICE, dtype=torch.float32)
            # reset hidden state
            init_hidden = model.encoder(init_firingrates)
            predicted_pc_fr, _ = model(velocity_inputs, init_hidden.unsqueeze(0))
            predicted_pc_fr = predicted_pc_fr.cpu().numpy()
            predicted_pc_frs.append(predicted_pc_fr)
            n_validation_batches += 1
    #now predict:
    predicted_pc_fr = np.concatenate(predicted_pc_frs, axis=0)  # (n_seq, seq_length, n_place_cells)
    predicted_pc_fr = predicted_pc_fr.reshape(n_validation_batches*options.batch_size*dataset.seq_length, options.Np)
    predicted_pos = gp_model.predict(predicted_pc_fr)
    decoding_errors_cm = np.linalg.norm(test_data.centroid_position.values[:predicted_pc_fr.shape[0]]-predicted_pos, axis = 1)*100
    return decoding_errors_cm
