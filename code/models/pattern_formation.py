'''The following script is meant to replicate Sorscher et al.'s (2023) pattern formation results.

Notation here is consistent with Sorscher et al.'s (2019) ICLR paper. '''

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

# Direct Lagrangian solution, gradient ascent

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd


def optimize_lagrangean(P, n_grid_cells, learning_rate=0.01, n_iterations=1000,  
                        nonnegative = True,
                        update_lambda=True, 
                        verbose=False):
    """
    Optimize grid cell activations G using Lagrangian gradient ascent/descent.
    
    Optimizes: L = Tr[G^T Σ G - Λ(G^T G - I)] where Σ = PP^T
    
    Args:
        P: Place cell activations (n_positions, n_place_cells)
        n_grid_cells: Number of grid cells
        learning_rate: Learning rate for optimization
        n_iterations: Number of optimization steps
        update_lambda: Whether to update Lagrange multipliers
        verbose: Whether to print progress
    
    Returns:
        dict: {'G': optimized grid cells, 'W': optimal readout weights, 'Lambda': Lagrange multipliers}
    """
    n_pos, n_placecells = P.shape
    
    # Initialize G with orthonormal columns
    G = np.random.randn(n_pos, n_grid_cells)*1e-8
    #U, s, Vt = svd(G, full_matrices=False)
    #G = U @ Vt
    
    # Initialize Lagrange multipliers
    Lambda = np.ones((n_grid_cells, n_grid_cells))*1e-8
    
    # Precompute Sigma = PP^T
    Sigma = P @ P.T
    #normalise Sigma
    Sigma = Sigma / np.linalg.norm(Sigma)
   
    #Sigma = (Sigma - Sigma.min()) / (Sigma.max() - Sigma.min())
    I = np.eye(n_grid_cells)
    
    # Optimization loop
    for i in range(n_iterations):
        # Compute gradients
        encode_grad = 2 * Sigma @ G 
        ortho_grad = 2 * G @ Lambda
        balance_scalar = 1 #np.abs(ortho_grad).max()/np.abs(encode_grad).max()
        grad_G = balance_scalar*encode_grad - ortho_grad
        
        # Update G (ascent or descent)
        G += learning_rate * (grad_G)
        
        if nonnegative:
            G = np.maximum(G,0)

        # Update Lagrange multipliers
        if update_lambda:
            grad_Lambda = -(G.T @ G - I)
            Lambda += learning_rate*(grad_Lambda - Lambda)
            Lambda = 0.5 * (Lambda + Lambda.T)  # Keep symmetric
   
        # Optional progress printing
        if verbose and i % 100 == 0:
            print(np.abs(ortho_grad).max(),np.abs(encode_grad).max(), G.max(),grad_G.max(), Sigma.max(), Lambda.max())
            lagrangian = np.trace(G.T @ Sigma @ G - Lambda @ (G.T @ G - I))
            constraint_violation = np.trace((G.T @ G - I).T @ (G.T @ G - I))
            print(f"Iter {i}: L = {lagrangian:.6f}, Constraint = {constraint_violation:.6f}")
        
        
    # Compute optimal W
    W = G.T @ P
    
    return {'G': G, 'W': W, 'Lambda': Lambda}



## Fast-fourier type solutions (code from Sorscher et al., 2023) ##

def fft_lagrangean_single(P, n_gridcells,learning_rate, n_iterations, 
                          verbose = False):
    '''
    Args:
        P: Place cell activations (n_positions, n_place_cells)
        n_grid_cells: Number of grid cells
        learning_rate: Learning rate for optimization
        n_iterations: Number of optimization steps
        update_lambda: Whether to update Lagrange multipliers
        verbose: Whether to print progress and plots
    
    Returns
        G: matrix corresponding to grid cell activations (n_positions, n_grid_cells)'''
    
    ratemap_size = np.sqrt(P.shape[0]) #assuming a square discretised environment.
    _, _, Ctilde = get_covariance_matrices(P)
    
    G = np.random.randn(n_gridcells,ratemap_size,ratemap_size)
    return G


def fft_lagrangean_population(P, n_gridcells, learning_rate, n_iterations):
    return None

def get_covariance_matrices(P, verbose = False):
    '''returns covariance matrices for a place cell activation matrix
    Arguments:
    ---------
    P: np.array()
        Place cell activations (n_positions, n_place_cells)
    verbose: bool
        Specify (true/false) whether to plot the covariance matrices.
    Returns:
    -------
    C: np.array()
        Place cell covariance matrix'''
    res = int(np.sqrt(P.shape[0])) #ratemap shape, assuming square discretised environment
    C = P@P.T
    Csquare = C.reshape(res,res,res,res)
    Cmean = np.zeros([res,res])
    n_additions = 0
    for i in range(res):
        for j in range(res):
            Cmean += np.roll(np.roll(Csquare[i,j], -i, axis=0), -j, axis=1)
            n_additions +=1
    Cmean = np.roll(np.roll(Cmean, res//2, axis=0), res//2, axis=1)/n_additions #fold the corners onto the center
    # Fourier transform
    Ctilde = np.fft.fft2(Cmean)
    Ctilde[0,0] = 0
    
    if verbose == True:
                
        ## plotting ##
        fig, ax = plt.subplots(1,4)
        ax[0].set(title = 'Example ratemap')
        ax[0].imshow(P[:,0].reshape(res,res))

        ax[1].set(title = r'C=$PP^T$=$\Sigma$')
        ax[1].imshow(C)

        ax[2].set(title ='Cmean')
        ax[2].imshow(Cmean)

        ax[3].set(title = 'Ctilde')
        width = 10
        idxs = np.arange(-width+1, width)
        x2, y2 = np.meshgrid(np.arange(2*width-1), np.arange(2*width-1))
        ax[3].scatter(x2,y2,c=np.abs(Ctilde)[idxs][:,idxs], s=600, cmap='Oranges', marker='s')
        ax[3].axis('square')
        fig.tight_layout()
    
    return C, Cmean, Ctilde

def convolve_with_C(g, Ctilde, ratemap_size):
    '''
    Convolves the input g with the kernel C
    '''
    gtilde = np.fft.fft2(g, [ratemap_size, ratemap_size])
    gconv = np.real(np.fft.ifft2(gtilde*Ctilde))
    gconv = np.roll(np.roll(gconv, ratemap_size//2+1, axis=1), ratemap_size//2+1, axis=2)
    
    return gconv
