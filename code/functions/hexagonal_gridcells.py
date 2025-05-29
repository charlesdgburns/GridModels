''' Minimal code library for a rectified cosine ('idealised grid cell') grid cell model.

Some code here is stolen from Tom George's RatInABox library.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from GridMaze.maze import representations as mr
from GridMaze.maze import plotting as mp


# %% Global variables

# %% main functions

def get_bin_mask_overlayed(bin_mask, n_neurons, n_modules):
    '''Returns a big matrix of potential ratemaps'''
    return None

def pixel_data2simple_maze(maze_name, binned_data):
    '''Treats the centre of each position as a position coordinate,
    then maps to the nearest node (or tower) and takes the mean for each node.'''
    
def get_grid_cell_module_on_maze(n_neurons, scale_in_metres, orientation, bin_mask, metres_per_bin):
    firingrates = get_grid_cell_module_from_bin_size(n_neurons,
                                                     scale_in_metres*metres_per_bin,
                                                     orientation,
                                                     map_size = bin_mask.shape[0])
    firingrates[:,(bin_mask!=1).flatten()] = np.nan
    labels, rates = mr.ratemap2simple_maze(firingrates,metres_per_bin)
    df = pd.DataFrame(data = rates)
    df['labels']=labels
    mean_df = df.groupby('labels').mean()
    return mean_df


def get_grid_cell_module_from_positions(n_neurons, scale, orientation, positions):
    """
    Compute idealised grid cell firing rates for a module with fixed spacing and orientation (uniformly random phase offsets).
    
    Parameters
    ----------
    n_neurons : int
        Number of grid cells from a given module.
    spacing : float
        Grid spacing (controls scale).
    orientation : float
        Orientation angle in radians.
    positions : ndarray
        Shape (n_positions, 2), each row is an [x, y] position.

    Returns
    -------
    firing_rates : ndarray
        Shape (n_positions, n_neurons), firing rate of each neuron at each position.
    """
    n_observations = positions.shape[0]

    # Grid axis angles
    angles = orientation + 2 * np.pi * np.array([0, 1/3, 2/3])  # 0, 120°, 240°
    basis_vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (3, 2)

    # Project positions → shape (n_observations, 3)
    projections = positions @ basis_vectors.T

    # Random 2D phase offsets for each neuron → (n_neurons, 2)
    offsets = np.random.uniform(0, scale, size=(n_neurons, 2))

    # Project each neuron's offset into grid space → (n_neurons, 3)
    offset_proj = offsets @ basis_vectors.T

    # Adjusted frequency
    adjusted_gridscale = scale / (2 / np.sqrt(3))  # match hexagonal spacing
    frequency = 2 * np.pi / adjusted_gridscale

    # Phase = position_proj - offset_proj → (n_obs, n_neurons, 3)
    phase = frequency * (
        projections[:, np.newaxis, :] - offset_proj[np.newaxis, :, :]
    )

    # Cosine response → normalize to [0, 1]
    response = (np.cos(phase) + 1) / 2  # shape: (n_observations, n_neurons, 3)

    # Combine responses from the 3 axes
    firing_rates = np.prod(response, axis=2)  # (n_observations, n_neurons)

    return firing_rates


def get_grid_cell_module_from_bin_size(n_neurons, spacing, orientation, map_size):
    '''Input a number of (phase shifted) cells requested for a given grid spacing.
    Returns ratemaps overlaid with a bin mask with 1s for valid locations.
    
    Parameters:
    ----------
    n_neurons: int()
        number of neurons with a given spacing and orientation
    
    Credit:
    -------
    This code is from Tom George's RatInABox library.'''
    
    gridscales = np.repeat(spacing, n_neurons)  # Grid scales for each neuron
    phase_offsets = np.random.uniform(low = 0, high = map_size, size = (n_neurons,2))  #uniformly distributed offsets
    width_ratio = 4 / (3 * np.sqrt(3)) #relevant ratio for rectification later.

    w = []
    for i in range(n_neurons):
        w1 = np.array([1.0, 0.0])
        w1 = rotate(w1, np.pi/6+orientation) # Apply orientation here, such that baseline has a peak due east
        w2 = rotate(w1, np.pi / 3)
        w3 = rotate(w1, 2 * np.pi / 3)
        w.append(np.array([w1, w2, w3]))
    w = np.array(w)
    pos = get_flattened_coords(map_size)
    origin = gridscales.reshape(-1, 1) * phase_offsets / (2 * np.pi)
    vecs = get_vectors_between(origin, pos)

    # Tile parameters for efficient calculation
    w1 = np.tile(np.expand_dims(w[:, 0, :], axis=1), reps=(1, pos.shape[0], 1))
    w2 = np.tile(np.expand_dims(w[:, 1, :], axis=1), reps=(1, pos.shape[0], 1))
    w3 = np.tile(np.expand_dims(w[:, 2, :], axis=1), reps=(1, pos.shape[0], 1))
    
    adjusted_gridscales = gridscales/(2/np.sqrt(3)) # account for sum of plane wave interaction radius
    tiled_gridscales = np.tile(np.expand_dims(adjusted_gridscales, axis=1), reps=(1, pos.shape[0]))

    phi_1 = ((2 * np.pi) / tiled_gridscales) * (vecs * w1).sum(axis=-1)
    phi_2 = ((2 * np.pi) / tiled_gridscales) * (vecs * w2).sum(axis=-1)
    phi_3 = ((2 * np.pi) / tiled_gridscales) * (vecs * w3).sum(axis=-1)

    firingrate = (1 / 3) * (np.cos(phi_1) + np.cos(phi_2) + np.cos(phi_3))

    # ... (rest of your code for firing rate calculation and plotting) ...

    #calculate the firing rate at the width fraction then shift, scale and rectify at the level
    a, b, c = np.array([1,0])@np.array([1,0]), np.array([np.cos(np.pi/3),np.sin(np.pi/3)])@np.array([1,0]), np.array([np.cos(np.pi/3),-np.sin(np.pi/3)])@np.array([1,0])
    firing_rate_at_full_width = (1 / 3) * (np.cos(np.pi*width_ratio*a) +
                                np.cos(np.pi*width_ratio*b) +
                                np.cos(np.pi*width_ratio*c))
    firing_rate_at_full_width = (1 / 3) * (2*np.cos(np.sqrt(3)*np.pi*width_ratio/2) + 1)
    firingrate -= firing_rate_at_full_width
    firingrate /= (1 - firing_rate_at_full_width)
    firingrate[firingrate < 0] = 0
    
    return firingrate


def get_grid_cell_modules(scales, orientations, offsets, map_size):
    '''Generate grid cell firing rate maps for combinations of parameters.
    
    Parameters:
    ----------
    scales : array-like
        Array of grid spacings to use
    orientations : array-like
        Array of orientations (in radians)
    offsets : array-like, shape (N, 2)
        Array of phase offsets, each row is [x, y]
    map_size : int
        Size of the square environment
        
    Returns:
    --------
    firingrates : ndarray
        Shape (n_scales * n_orientations * n_offsets, map_size^2)
        Each row is a flattened ratemap for a parameter combination
    params : dict
        Dictionary containing the parameter values used for each ratemap
    '''
    width_ratio = 4 / (3 * np.sqrt(3))
    
    # Get all combinations of parameters
    n_scales = len(scales)
    n_orientations = len(orientations)
    n_offsets = len(offsets)
    n_maps = n_scales * n_orientations * n_offsets
    
    # Store parameters for each ratemap
    params = {
        'scale': np.repeat(scales, n_orientations * n_offsets),
        'orientation': np.tile(np.repeat(orientations, n_offsets), n_scales),
        'offset_x': np.tile(offsets[:, 0], n_scales * n_orientations),
        'offset_y': np.tile(offsets[:, 1], n_scales * n_orientations)
    }
    
    # Initialize arrays
    pos = get_flattened_coords(map_size)
    w = np.zeros((n_maps, 3, 2))
    
    # Calculate basis vectors for all maps at once
    for i in range(n_maps):
        w1 = np.array([1.0, 0.0])
        w1 = rotate(w1, np.pi/6 + params['orientation'][i])
        w2 = rotate(w1, np.pi/3)
        w3 = rotate(w1, 2*np.pi/3)
        w[i] = np.stack([w1, w2, w3])
    
    # Calculate origins and vectors
    origins = params['scale'].reshape(-1, 1) * np.stack([params['offset_x'], params['offset_y']], axis=1) / (2 * np.pi)
    vecs = get_vectors_between(origins, pos)
    
    # Tile parameters for efficient calculation
    w1 = np.tile(np.expand_dims(w[:, 0, :], axis=1), reps=(1, pos.shape[0], 1))
    w2 = np.tile(np.expand_dims(w[:, 1, :], axis=1), reps=(1, pos.shape[0], 1))
    w3 = np.tile(np.expand_dims(w[:, 2, :], axis=1), reps=(1, pos.shape[0], 1))
    
    # Adjust scales and tile
    adjusted_scales = params['scale']/(2/np.sqrt(3))
    tiled_scales = np.tile(np.expand_dims(adjusted_scales, axis=1), reps=(1, pos.shape[0]))
    
    # Calculate phases
    phi_1 = ((2 * np.pi) / tiled_scales) * (vecs * w1).sum(axis=-1)
    phi_2 = ((2 * np.pi) / tiled_scales) * (vecs * w2).sum(axis=-1)
    phi_3 = ((2 * np.pi) / tiled_scales) * (vecs * w3).sum(axis=-1)
    
    # Calculate firing rates
    firingrates = (1/3) * (np.cos(phi_1) + np.cos(phi_2) + np.cos(phi_3))
    
    # Rectify firing rates
    a = np.array([1,0])@np.array([1,0])
    b = np.array([np.cos(np.pi/3),np.sin(np.pi/3)])@np.array([1,0])
    c = np.array([np.cos(np.pi/3),-np.sin(np.pi/3)])@np.array([1,0])
    
    firing_rate_at_full_width = (1/3) * (2*np.cos(np.sqrt(3)*np.pi*width_ratio/2) + 1)
    firingrates -= firing_rate_at_full_width
    firingrates /= (1 - firing_rate_at_full_width)
    firingrates[firingrates < 0] = 0
    
    return firingrates, params
# %% utility functions
def rotate(vector, theta):
    """Rotates a vector anticlockwise by angle theta."""
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    vector_new = np.dot(R, vector)
    return vector_new

def get_vectors_between(pos1, pos2):
    """Calculates vectors between two sets of positions."""
    pos1_ = pos1.reshape(-1, 1, pos1.shape[-1])
    pos2_ = pos2.reshape(1, -1, pos2.shape[-1])
    pos1 = np.repeat(pos1_, pos2_.shape[1], axis=1)
    pos2 = np.repeat(pos2_, pos1_.shape[0], axis=0)
    vectors = pos1 - pos2
    return vectors

def get_flattened_coords(N):
    """Generates flattened coordinates for a square meshgrid."""
    x = np.arange(N)
    y = np.arange(N)
    xv, yv = np.meshgrid(x, y)
    return np.stack((xv.flatten(), yv.flatten()), axis=1)
