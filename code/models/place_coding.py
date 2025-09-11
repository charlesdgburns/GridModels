'''Here we place some scripts to generate place codes using different methods (kernels) and metrics.'''



import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist
from GridModels.code.maze import representations as mr

def get_place_code(positions,
                   sigma: float,
                   n_place_cells = 512,
                   bin_size:float = 0.02,
                   method:str = 'gaussian', #'gaussian','diff_of_gaussians',
                   distance_metric:str = 'euclidean', #'euclidean','geodesic',
                   dog_ratio:float = 3/2,
                   allow_diagonal = True,
                   random_seed:float = 42):
    '''
    Returns 
    -------
    P: np.array() (n_pos,n_place_cells)
        array of place cell activatotns for each discretised (binned) position.
    pixel_maze: networkx graph object
        graph built from discretsed binned positions.
    bin_mask: np.array() (n_xbins, n_ybins) bool
        TRUE/FALSE for whether each bin has been visited
    
    '''
    #set seed for reproducibility
    np.random.seed(random_seed)
    
    pixel_maze, bin_mask = mr.get_pixel_maze(positions, 
                                             pixel_size = bin_size,
                                             allow_diagonal=allow_diagonal)
    
    if distance_metric == 'geodesic':
        nodes = [x for x in pixel_maze.nodes()]
        A = nx.to_scipy_sparse_array(pixel_maze, nodelist=nodes, weight="weight", format="csr")
        D = shortest_path(A, directed=G.is_directed(), unweighted=False)  # D[i, j] is distanceplt.imshow(D)
    elif distance_metric == 'euclidean':
        centers = np.array([x for x in nx.get_node_attributes(pixel_maze,'position').values()])
        D = cdist(centers,centers)
  
    if method == 'gaussian':
        P = gaussian(D, sigma=sigma)
    elif method == 'diff_of_gaussians':
        P = diff_of_gaussians(D, sigma, dog_ratio)
   
    subset_idx = np.random.choice(np.arange(len(P)), n_place_cells)
    P = P[:,subset_idx] #(n_pos, n_place_cells)
    return P, pixel_maze, bin_mask


# definition of kernels
def gaussian(distance, sigma:float):
    return np.exp(-(distance**2)/(2*sigma**2))

def diff_of_gaussians(distance, sigma:float, dog_ratio:float):
    small_gaussian = np.exp(-(distance**2)/(2*sigma**2))
    big_gaussian = (1/dog_ratio**2)*np.exp(-(distance**2)/(2*(dog_ratio*sigma)**2))
    normalising_constant = dog_ratio**2/(dog_ratio**2-1)
    return normalising_constant*(small_gaussian-big_gaussian)
