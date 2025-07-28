"""Code to fit lattice to autocorrelograms resulting in extraction of grid parameters.
    Assumes autocorrelograms have already been computed via spatial.py

We attempt following methods in https://www.nature.com/articles/nn.3450#MOESM29 
implemented in using PyTorch.
"""

# %% Setup
import math 
import numpy as np
import torch
import torch.nn.functional as F
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

## Local imports
from GridMaze.analysis.cluster_tuning import spatial
from GridMaze.analysis.core.folds import kfold_leave_one_out
from experiment.code.GridModels.code.gridcell_models import hexagonal_gridcells as hex

# %% Global Variables
#ratemap keyword arguments:



# %% Fitting a rectified cosine model to a cell's ratemap.

# TOP LEVEL FUNCTION:

rm_kwargs = {
    "x_size": 0.02,
    "y_size": 0.02,
    "smooth_SD": 0.04,
    "x_range": (0.05, 1.35),
    "y_range": (0.05, 1.35),
    "nan_unvisited": True,
}

def fit_cv_hexagonal_model(pos, spikes, n_folds = 5, rm_kwargs= rm_kwargs, device= DEVICE):
    '''From spikes and position data, computes a 2D ratemap and fits a hexagonal grid cell model to it.
    Will also return a cross-validated ratemap score on n_fold leave-one-out train->fit and test->correlation.'''    
    # generate firing rates using full data
    full_ratemap, binx, biny = spatial.get_2D_ratemap(spikes, pos, **rm_kwargs)
    full_out = fit_hexagonal_model2ratemap(full_ratemap, bins2metres = rm_kwargs['x_size'],
                                        device = device,)

    ## get cross-validated correlation score (check agains overfitting)
    cv_correlations = []
    for train_pos, train_spikes, test_pos, test_spikes in kfold_leave_one_out(pos, spikes, n_folds):
        train_ratemap, binx, biny = spatial.get_2D_ratemap(train_spikes, train_pos, **rm_kwargs)
        test_ratemap, _, _ = spatial.get_2D_ratemap(test_spikes, test_pos, **rm_kwargs)
        out = fit_hexagonal_model2ratemap(train_ratemap, bins2metres = rm_kwargs['x_size'], device='cuda')
        best_fit_ratemap = out['fitted_ratemap']
        overlap_mask = np.isfinite(best_fit_ratemap) & np.isfinite(test_ratemap)
        if np.any(overlap_mask):
            tensor_target = torch.tensor(test_ratemap[overlap_mask], device=device)
            tensor_batch = torch.tensor(best_fit_ratemap[overlap_mask], device=device).unsqueeze(0)
            corr = compute_batch_correlation(tensor_target, tensor_batch).item()
        else:
            corr = np.nan
        cv_correlations.append(corr)
        
    full_out[f'cv_corr'] = np.mean(cv_correlations)
    return full_out

def fit_hexagonal_model2ratemap(ratemap, bins2metres, device = DEVICE):
    '''Takes a ratemap and fits an idealised hexagonal grid cell model to it.
    Parameters:
    -----------
    ratemap: array-like (map_size, map_size)
        The ratemap of the grid cell to fit the model to. Can be numpy or torch.tensor()

    Returns:
    --------
    output: dict
        A dictionary containing the fitted parameters of the hexagonal grid cell model.
        Keys include 'scale', 'orientation','offset', 'full_corr', 'cv_corr'.
    
    '''
    map_size = ratemap.shape[0]
    orientations =np.linspace(-np.pi/6,np.pi/6,30) # np.array([0.0])    #from -15 to +15 degrees in radians
    scales = np.linspace(5,map_size,map_size//1) # np.array([1.0])              #scale in ratemap bins
    offsets = get_grid_offsets(map_size//2, map_size) #np.array([[0.0,0.0]])     #offsets in ratemap bin coordinates
    
    #GPU RAM management here: loop over a batch of scales, for all offsets and orientations:
    #to minimise GPU RAM used, we consider only positions with non-nan rates:
    max_n_scales = get_max_n_scales(len(offsets), len(orientations), map_size)
    n_batches = len(scales) // max_n_scales + 1
    scales_batches = np.array_split(scales, n_batches)
    for i,scale in enumerate(scales_batches):
        non_nan_indices = np.argwhere(ratemap.flatten() >=0)
        position_array = hex.get_flattened_coords(map_size)
        positions = position_array[non_nan_indices].squeeze(1) #(n_valid_positions,2)
        firingrates, params = hex.get_fast_hexagonal_rates(positions,scale,orientations,offsets, device = device)
        
        full_ratemap = torch.tensor(ratemap.flatten()[non_nan_indices]).flatten().to(device)
        corr = compute_batch_correlation(full_ratemap, firingrates)
        best_idx = torch.argmax(corr).item()
        
        best_fit_ratemap, _ = hex.get_fast_hexagonal_rates(position_array, 
                                            [params[best_idx]['scale']],
                                            [params[best_idx]['orientation']],
                                            [params[best_idx]['offset']],
                                            device = device)
        best_fit_ratemap = best_fit_ratemap[0]
        best_fit_ratemap[np.isnan(ratemap.flatten())] = np.nan #put nans back in
        
        output = {'scale': params[best_idx]['scale']*bins2metres,
                'orientation': np.rad2deg(params[best_idx]['orientation']),
                'offset': params[best_idx]['offset']*bins2metres,
                'correlation': corr[best_idx].item(),
                'fitted_ratemap': best_fit_ratemap.cpu().reshape(map_size, map_size).numpy()}

        
        if i == 0:
            best_output = output.copy()
        else:
            if output['correlation'] > best_output['correlation']:
                best_output = output.copy()
        ## GPU RAM Management:
        del firingrates, params, full_ratemap, corr, position_array
        torch.cuda.empty_cache()  # Clear GPU memory cache
    # Return the best output
    return best_output


### Utility functions for fitting hexagonal model

def summarise_cuda_memory():
    torch.cuda.empty_cache()  # Clear GPU memory cache
    torch.cuda.empty_cache()  # Clear GPU memory cache
    
    #Get the total memory of the GPU
    total_memory = torch.cuda.get_device_properties(0).total_memory

    # Get the currently allocated memory
    allocated_memory = torch.cuda.memory_allocated(0)

    # Get the currently reserved memory
    reserved_memory = torch.cuda.memory_reserved(0)

    print(f"Total memory: {total_memory / (1024**3):.2f} GB")
    print(f"Allocated memory: {allocated_memory / (1024**3):.2f} GB")
    print(f"Reserved memory: {reserved_memory / (1024**3):.2f} GB")
    return None

def get_max_n_scales(n_offsets, n_orientations, map_size):
    """Calculate how many scales can be tried given the map size, n_offsets, and n_orientations.
    Computes based on how much memory is available on the GPU.
    Args:
        n_offsets: int, number of offsets to try
        n_orientations: int, number of orientations to try
        map_size: int, size of the map (assumed square)"""
    # Get the total memory of the GPU
    total_memory = torch.cuda.get_device_properties(0).total_memory

    # Get the currently allocated memory
    allocated_memory = torch.cuda.memory_allocated(0)

    # Calculate available memory
    available_memory = total_memory - allocated_memory

    # Estimate memory usage per scale
    # Assuming each float32 takes 4 bytes, and we have:
    # - n_offsets * n_orientations firing rates per position
    # - map_size^2 positions in the ratemap
    memory_per_scale = (3*n_offsets*2 * n_orientations * map_size**2 *2* 4)  # in bytes

    # Calculate maximum number of scales that can be tried
    max_scales = available_memory // memory_per_scale

    return int(max_scales)

def get_grid_offsets(n_per_side, L):
    """Generate evenly spaced offsets covering a square.

    Args:
        n_per_side: int, number of points along each dimension
        L: float, length of square in same units as grid spacing

    Returns:
        Array of shape (n_per_side^2, 2) containing [x,y] offset coordinates
    """
    # Create evenly spaced points along each dimension
    x = np.linspace(0, L, n_per_side, endpoint=False)
    y = np.linspace(0, L, n_per_side, endpoint=False)

    # Create meshgrid and reshape to (n_points, 2)
    xx, yy = np.meshgrid(x, y)
    offsets = np.stack([xx.ravel(), yy.ravel()]).T

    return offsets 


def compute_batch_correlation(target_values, batch_values):
  """
  Computes the correlation between a single map and a batch of other maps quickly via GPU.

  Args:
    target_map: A PyTorch tensor of shape (n_positions) representing the target map.
    batch_maps: A PyTorch tensor of shape (n_combinations, n_positions) representing the batch of maps.

  Returns:
    A PyTorch tensor of shape (n_combinations,) containing the correlation coefficients.
  """

  # Normalize the target map
  target_map_norm = (target_values - target_values.mean()) / target_values.std()

  # Normalize the batch maps
  batch_maps_norm = (batch_values - batch_values.mean(dim=(1), keepdim=True)) / batch_values.std(dim=(1), keepdim=True)

  # Compute the correlation for each map in the batch
  correlation_coefficients = torch.nn.functional.cosine_similarity(target_map_norm.unsqueeze(0),
                                                                 batch_maps_norm.flatten(1),
                                                                 dim=1)
  return correlation_coefficients

def compute_batch_CV(target_values, batch_values):
  """
  Computes the coefficient of variation (as RMSE/mean) between a single map and a batch of other maps quickly via GPU.

  Args:
    target_map: A PyTorch tensor of shape (n_positions) representing the target map.
    batch_maps: A PyTorch tensor of shape (n_combinations, n_positions) representing the batch of maps.

  Returns:
    A PyTorch tensor of shape (n_combinations,) containing the root mean squared difference.
  """

  # Normalize the target map
  target_map_norm = target_values/ target_values.max()
  # Normalize the batch maps
  batch_maps_norm = batch_values / batch_values.max(dim=(1), keepdim=True).values
  
  squared_error = (target_map_norm.unsqueeze(0) - batch_maps_norm.flatten(1))**2
  
  # Compute the correlation for each map in the batch
  coefficient_of_variation = torch.sqrt(squared_error.mean(dim=1,keepdim=True))/target_map_norm.mean()
  return coefficient_of_variation

def compute_batch_SSIM(target_values, batch_values):
    """
    Computes the Structural Similarity Index Measure between a single map and a batch of other maps quickly via GPU.

    Args:
    -----
    target_map: A PyTorch tensor of shape (n_positions) representing the target map.
    batch_maps: A PyTorch tensor of shape (n_combinations, n_positions) representing the batch of maps.

    Returns:
    --------
    A PyTorch tensor of shape (n_combinations,) containing the Structured Similarity Index Measure.
    """
    ##IMPLEMENTING WIKIPEDIA DEFINITION OF SSIM:
    #assume float32 inputs, so 32 bits per pixel.
    L = 2^32 - 1 # dynamic range of float32 pixel values
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2

    # Normalize the maps so they range from 0 to 1
    target_map_norm = target_values/ target_values.max()
    batch_maps_norm = batch_values / batch_values.max(dim=(1), keepdim=True).values
    covariance = (target_map_norm - target_map_norm.mean()).unsqueeze(0)*(batch_maps_norm - batch_maps_norm.mean(dim=1,keepdim=True))
    covariance = covariance.mean(dim=1) #(n_maps)

    # Compute the SSIM numerator and denominator
    numerator = (2*target_map_norm.mean() * batch_maps_norm.mean(dim=1) + c1)*(2*covariance + c2)
    denominator = (target_map_norm.var() + batch_maps_norm.var(dim=1) + c1) * (target_map_norm.mean()**2 + batch_maps_norm.mean(dim=1)**2 + c2)

    ssim = numerator / denominator
    return ssim

def kfold_leave_one_out(pos, spikes, k):
    """
    Splits data into k contiguous chunks and yields (train_pos, train_spikes, test_pos, test_spikes) for each fold.
    No shuffling; preserves order within chunk.

    Args:
        pos: np.ndarray of shape (N, 2)
        spikes: np.ndarray of shape (N,)
        k: int, number of folds

    Yields:
        train_pos, train_spikes, test_pos, test_spikes for each fold
    """
    n = len(spikes)
    fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    indices = np.arange(n)
    current = 0
    splits = []
    for fold_size in fold_sizes:
        splits.append(indices[current:current + fold_size])
        current += fold_size

    for i in range(k):
        test_idx = splits[i]
        train_idx = np.concatenate([splits[j] for j in range(k) if j != i])
        yield pos[train_idx], spikes[train_idx], pos[test_idx], spikes[test_idx]








# %% Utility functions


def standardise_basis_vectors(basis_1,basis_2):
  if len(basis_1.shape)==1:
    basis_1 = basis_1.unsqueeze(1)
    basis_2 = basis_2.unsqueeze(1)
  ## we transform a small grid by the basis vectors, so getting a lattice with nine points:
  std_basis_coords = basis_vectors2lattice_coords(basis_1, basis_2, meshgrid_size=3)
  #the central point will be at (0,0) so we set that to (-1,-1) to not get chosen:
  std_basis_coords[:,4] = torch.tensor([[-1],[-1]])
  #then compute angles ranging from -pi to pi:
  angles = torch.arctan2(std_basis_coords[1],std_basis_coords[0])
  # smallest absolute angle will be due east
  east_vector_idx = torch.argmin(angles.abs())
  # north east angle will be at least 45 degrees greater than it:
  putative_NE_vector_idx = torch.argwhere((angles.flatten()>angles[east_vector_idx]) & (angles.flatten()>(angles[east_vector_idx]+np.pi/4)))
  smallest_putative_NE_angle_idx = torch.argmin(angles[putative_NE_vector_idx])
  north_east_vector_idx = putative_NE_vector_idx[smallest_putative_NE_angle_idx][0]

  b1 = std_basis_coords[:,east_vector_idx].squeeze() #shape (2)
  b2 = std_basis_coords[:,north_east_vector_idx].squeeze() #shape (2)
  
  return b1, b2
  #then to get a standardised measure of global orientation and angle between vectors:
  ## we want to choose the vectors pointing roughly due east and north-east.
  ## that is, the vector with smallest angle (due east) and then the vector rotated to the left of it.
  ## However, we have a restriction that the second vector must be at least 45 degrees, due to inclusion of more than just 6 peaks in the lattice.
    #(the sum of standard basis vectors would introduce another with wrong angle)


def basis_vectors2lattice_coords(basis_1, basis_2, meshgrid_size=7):
    '''Returns coordinates of a lattice generated by basis vectors.
    
    Parameters
    ----------
    basis_1: torch.tensor() #(2,n_basis_vectors)
    basis_2: torch.tensor() #(2,n_basis_vectors)
        tensorised basis vectors, such that n_basis_vectors can also be 1.

    Returns
    -------
    lattice_coords: torch.tensor() #(2,n_lattice,n_basis_vectors)
        where n_lattice is the nuber of coordinates, equal to meshgrid_size^2
    '''
    if len(basis_1.shape) ==1:
        n_basis_vectors = 1
        basis_1 = basis_1.unsqueeze(1)
        basis_2 = basis_2.unsqueeze(1)
    else:
        n_basis_vectors = basis_1.shape[1]
    # Generate the range of coefficients for linear combinations
    coeff_range = torch.arange(-int(meshgrid_size/2),np.round(meshgrid_size/2)) #slightly hacky, but int() rounds down and round() rounds up
    # Generate the grid of coefficients
    c1, c2 = torch.meshgrid(coeff_range, coeff_range)
    # Initialize the final_result tensor
    batch_of_coords = torch.zeros( 2,meshgrid_size*meshgrid_size, n_basis_vectors)

    # Iterate through the dimensions of the meshgrids
    for i in range(meshgrid_size):
        for j in range(meshgrid_size):
            # Calculate the linear index for the current point
            linear_index = i * meshgrid_size + j

            # Multiply the meshgrid values with the basis vectors
            coords = c1[i, j] * basis_1 + c2[i, j] * basis_2
            batch_of_coords[:,linear_index, :] = coords
    return batch_of_coords


def params2basis_vectors(params):
    '''Returns basis vectors from a parameters dictionary with 1, l2, theta, and phi'''
    #hard-coded trigonometry
    l1 = params['l1']
    l2=params['l2']
    theta=params['theta']
    phi=params['phi']
    basis_1 = torch.stack([l1*torch.cos(phi),
                           -l1*torch.sin(phi)])
    basis_2 = torch.stack([l2*torch.cos(theta-phi),
                           l2*torch.sin(theta-phi)])
    return basis_1, basis_2

def basis_vectors2params(b1, b2):#
  '''Converts two basis vectors into dictionary with l1, l2, theta, and phi parameters'''
  if len(b1.shape)>1:
    b1 = b1.squeeze()
    b2 = b2.squeeze()
  #do some hard-coded trigonometry
  length_b1 = torch.sqrt(b1[0]**2+b1[1]**2) #by pythagorean theorem, because torch.norm is cheating
  length_b2 = torch.sqrt(b2[0]**2+b2[1]**2) #by pythagorean theorem
  theta = torch.arccos((torch.dot(b1,b2))/(length_b1*length_b2)) #angle between vectors
  phi = torch.arctan2(b1[1],b2[0]) #angle of basis_1 with x-axis

  params = {'l1':length_b1,
            'l2':length_b2,
            'theta':theta,
            'phi':phi}
  return params


def get_distance_to_inner_peaks(peak_coords, centre_coords):
    centred_peak_coords = peak_coords - centre_coords
    peak_dist_to_centre = np.hypot(centred_peak_coords[0], centred_peak_coords[1])
    median_dist_to_inner = peak_dist_to_centre.sort().values[1:6].median()
    return median_dist_to_inner

def circle_mask(autocorr, relative_radius=1, in_val=1.0, out_val=0.0):
    """Calculating the grid scores with different radius.
    Size = [n_bins_x,n_bins_y],
    radius = int proportional to number of bins"""
    size = autocorr.shape
    radius = int(round(autocorr.shape[0]/2))*relative_radius
    sz = [math.floor(size[0] / 2), math.floor(size[1] / 2)]
    x = np.linspace(-sz[0], sz[1], size[1])
    x = np.expand_dims(x, 0)
    x = x.repeat(size[0], 0)
    y = np.linspace(-sz[0], sz[1], size[1])
    y = np.expand_dims(y, 1)
    y = y.repeat(size[1], 1)
    z = np.sqrt(x**2 + y**2)
    z = np.less_equal(z, radius)
    vfunc = np.vectorize(lambda b: b and in_val or out_val)
    return vfunc(z)

def rotation_matrix(phi:torch.tensor):
    s = torch.sin(phi)
    c = torch.cos(phi)
    return torch.stack([torch.stack([c,-s]),
                        torch.stack([s,c])]) 
    
def shear_matrix(shear_lambda:torch.tensor):
    return torch.stack([torch.tensor([1.0,shear_lambda]),
                        torch.tensor([0.0, 1.0])])


def fast_gaussian_blur(image_tensor, kernel_size=15, sigma=1.5):
    """
    Applies a Gaussian blur to a batch of images using PyTorch.

    Args:
        image_tensor: A PyTorch tensor of shape (n_images, image_size_x, image_size_y).
        kernel_size: The size of the Gaussian kernel.
        sigma: The standard deviation of the Gaussian kernel.

    Returns:
        A PyTorch tensor of the same shape as image_tensor with the Gaussian blur applied.
    """

    # Create a Gaussian kernel
    kernel = _gaussian_kernel(kernel_size, sigma)
    kernel = kernel.to(image_tensor.device) # Move kernel to the same device as the image tensor
    
    # Apply the blur using convolution
    blurred_images = F.conv2d(image_tensor.unsqueeze(1), kernel.unsqueeze(0), padding=kernel_size // 2, groups=1)
    
    return blurred_images.squeeze(1)

def _gaussian_kernel(kernel_size, sigma):
  """
  Generates a 2D Gaussian kernel.

  Args:
    kernel_size: The size of the kernel (must be odd).
    sigma: The standard deviation of the Gaussian.

  Returns:
    A 2D Gaussian kernel as a PyTorch tensor.
  """
  x_cord = torch.arange(kernel_size)
  x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
  y_grid = x_grid.t()
  xy_grid = torch.stack([x_grid, y_grid], dim=-1)

  mean = (kernel_size - 1)/2.
  variance = sigma**2.

  gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2 * variance))
  gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
  
  return gaussian_kernel.float().unsqueeze(0) # Add a channel dimension for convolution
