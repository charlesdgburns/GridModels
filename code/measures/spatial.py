"""Library to plotting place direction heatmaps"""

# %% Imports
import math
import numpy as np

from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# %% Global Variables
FRAME_RATE = 60  # Hz

# %% Functions

def get_2D_ratemap(
    spikes: np.ndarray,
    pos: np.ndarray,
    x_size: float = 0.02,  # Bin size in meters
    y_size: float = 0.02,  # Bin size in meters
    smooth_SD: float = 0.04,  # Smoothing window (standard deviation) in meters
    x_range=None,  #
    y_range=None,  #
    nan_unvisited=True,
):
    """
    Parameters
    ----------
    spikes: ndarray (n,)
        Number of spikes that occurred at each time step.
    pos: ndarray (n, 2)
        x, y coordinates representing the position of the animal when spikes occurred.
    x_size: float
        Bin size in meters for the x dimension.
    y_size: float
        Bin size in meters for the y dimension.
    smooth_SD: float
        Standard deviation of the Gaussian filter in meters. If 0, no smoothing will be applied.

    Returns
    -------
    h: ndarray (nybins, nxbins)
        Firing rate (Hz) falling on each bin through the recorded session. nybins is the number of bins in the y axis,
        nxbins is the number of bins in the x axis.
    binx: ndarray (nxbins +1,)
        Bin limits of the ratemap on the x axis.
    biny: ndarray (nybins +1,)
        Bin limits of the ratemap on the y axis.
    """
    x, y = pos[:, 0], pos[:, 1]  # Extract x and y coordinates

    # Determine the number of bins based on the range of x, y data and desired bin size
    if x_range is None:
        x_min, x_max = np.min(x), np.max(x)
    else:
        x_min, x_max = x_range
    if y_range is None:
        y_min, y_max = np.min(y), np.max(y)
    else:
        y_min, y_max = y_range

    # Calculate the number of bins based on the data range and the desired bin size
    nxbins = int((x_max - x_min) / x_size)
    nybins = int((y_max - y_min) / y_size)

    # Create a 2D histogram of the number of spikes (weighted histogram)
    spike_hist, binx, biny = np.histogram2d(
        x, y, bins=[nxbins, nybins], weights=spikes, range=[[x_min, x_max], [y_min, y_max]]
    )

    # Calculate the time spent in each bin (occupancy)
    # Time per frame is 1 / FRAME_RATE
    time_per_frame = 1 / FRAME_RATE
    # Create a 2D histogram of the number of frames spent in each bin
    occupancy, _, _ = np.histogram2d(x, y, bins=[nxbins, nybins], range=[[x_min, x_max], [y_min, y_max]])
    occupancy *= time_per_frame  # Convert frame count to time spent in each bin (seconds)

    # Compute the firing rate (spikes per second) for each bin
    # Avoid division by zero by only dividing where occupancy > 0
    h = np.zeros_like(spike_hist)
    valid_bins = occupancy > 0
    h[valid_bins] = spike_hist[valid_bins] / occupancy[valid_bins]

    # Apply Gaussian smoothing if smooth_SD > 0
    if smooth_SD > 0:
        #smoothing needs to be corrected for smoothing over unvisited bins:
        weights = h.copy()
        weights[occupancy!=0] = 1 ## make constant in unvisited bins
        # Convert smoothing window (SD) from meters to bin units
        sigma_x = smooth_SD / x_size
        sigma_y = smooth_SD / y_size
        h = gaussian_filter(h, sigma=[sigma_x, sigma_y])
        weights = gaussian_filter(weights, sigma=[sigma_x, sigma_y])
        h = h/weights
        
    if nan_unvisited:
        # Set bins to np.nan if they were not visited
        h[occupancy == 0] = np.nan

    # Transpose to change row-column coordinates to positions
    return h.T, binx, biny

### ratemap function for multiple spikes in a session:
 

def get_2D_ratemaps(
    spikes: np.ndarray,
    pos: np.ndarray,
    x_size: float = 0.02,  # Bin size in meters
    y_size: float = 0.02,  # Bin size in meters
    smooth_SD: float = 0.04,  # Smoothing window (standard deviation) in meters
    x_range=None,
    y_range=None,
    frame_rate: float = FRAME_RATE,  # Added frame_rate parameter
):
    """
    Parameters
    ----------
    spikes: ndarray (n_spikes, m_units)
        Number of spikes that occurred at each time step for m units.
    pos: ndarray (n, 2)
        x, y coordinates representing the position of the animal when spikes occurred.
    x_size: float
        Bin size in meters for the x dimension.
    y_size: float
        Bin size in meters for the y dimension.
    smooth_SD: float
        Standard deviation of the Gaussian filter in meters. If 0, no smoothing will be applied.
    x_range: tuple, optional
        (x_min, x_max) range for the x axis. If None, computed from data.
    y_range: tuple, optional
        (y_min, y_max) range for the y axis. If None, computed from data.
    frame_rate: float, optional
        Frame rate of recording in Hz. Default is 30.

    Returns
    -------
    h: ndarray (nybins, nxbins, m)
        Firing rate maps (Hz) for each unit. nybins and nxbins represent the spatial bins.
    binx: ndarray (nxbins + 1,)
        Bin limits of the ratemap on the x axis.
    biny: ndarray (nybins + 1,)
        Bin limits of the ratemap on the y axis.
    """
    x, y = pos[:, 0], pos[:, 1]  # Extract x and y coordinates

    # Determine the number of bins based on the range of x, y data and desired bin size
    if x_range is None:
        x_min, x_max = np.min(x), np.max(x)
    else:
        x_min, x_max = x_range
    if y_range is None:
        y_min, y_max = np.min(y), np.max(y)
    else:
        y_min, y_max = y_range

    # Calculate the number of bins
    nxbins = int((x_max - x_min) / x_size)
    nybins = int((y_max - y_min) / y_size)

    # Bin positions into a 2D histogram for occupancy (time spent in each bin)
    occupancy, binx, biny = np.histogram2d(
        x, y, bins=[nxbins, nybins], range=[[x_min, x_max], [y_min, y_max]]
    )
    occupancy *= 1 / frame_rate  # Convert frames to time in seconds
    
    # Compute spike histograms for all units
    # First make sure the dimensions are as expected
    if len(spikes.shape) == 1:
        spikes = spikes.reshape(len(spikes), 1)
    
    hists = []
    for unit_spikes in spikes.T:
        spike_hist, _, _ = np.histogram2d(
            x, y, bins=[nxbins, nybins], weights=unit_spikes, range=[[x_min, x_max], [y_min, y_max]]
        )
        hists.append(spike_hist)
    hists = np.stack(hists, axis=-1)  # Shape: (nxbins, nybins, m)

    # Compute firing rates by dividing spike histograms by occupancy
    valid_bins = occupancy > 0
    h = np.zeros_like(hists)
    
    # Fix broadcasting issue - expand occupancy to match hists dimensions
    for i in range(hists.shape[-1]):
        h[..., i][valid_bins] = hists[..., i][valid_bins] / occupancy[valid_bins]

    # Apply Gaussian smoothing if smooth_SD > 0
    if smooth_SD > 0:        
        # Convert smoothing window (SD) from meters to bin units
        sigma_x = smooth_SD / x_size
        sigma_y = smooth_SD / y_size
        
        # Apply smoothing to each unit separately
        for i in range(h.shape[-1]):
            # Create weights for normalization (1 where we have data, 0 elsewhere)
            weights = np.ones_like(h[..., i])
            weights[occupancy == 0] = 0
            
            # Apply smoothing
            h_smoothed = gaussian_filter(h[..., i], sigma=[sigma_x, sigma_y])
            weights_smoothed = gaussian_filter(weights, sigma=[sigma_x, sigma_y])
            
            # Normalize by smoothed weights to avoid edge effects
            valid_smooth = weights_smoothed > 0
            h[..., i][valid_smooth] = h_smoothed[valid_smooth] / weights_smoothed[valid_smooth]
    
    # Set bins with no occupancy to NaN
    for i in range(h.shape[-1]):
        h[..., i][occupancy == 0] = np.nan
    
    # Transpose to change row-column coordinates to positions
    return h.T, binx, biny


def autoCorr2D(A, nodwell, tol=1e-10):
    """
    Performs a spatial autocorrelation on the array A
    Parameters
    ----------
    A : array_like
        Either 2 or 3D. In the former it is simply the binned up ratemap
        where the two dimensions correspond to x and y.
        If 3D then the first two dimensions are x
        and y and the third (last dimension) is 'stack' of ratemaps
    nodwell : array_like
        A boolean array corresponding the bins in the ratemap that
        weren't visited. See Notes below.
    tol : float, optional
        Values below this are set to zero to deal with v small values
        thrown up by the fft. Default 1e-10
    Returns
    -------
    sac : array_like
        The spatial autocorrelation in the relevant dimensionality
    Notes
    -----
    The nodwell input can usually be generated by:
    >>> nodwell = ~np.isfinite(A)
    """

    assert np.ndim(A) == 2
    m, n = np.shape(A)
    o = 1
    x = np.reshape(A, (m, n, o))
    nodwell = np.reshape(nodwell, (m, n, o))
    x[nodwell] = 0
    # [Step 1] Obtain FFTs of x, the sum of squares and bins visited
    Fx = np.fft.fft(np.fft.fft(x, 2 * m - 1, axis=0), 2 * n - 1, axis=1)
    FsumOfSquares_x = np.fft.fft(np.fft.fft(np.power(x, 2), 2 * m - 1, axis=0), 2 * n - 1, axis=1)
    Fn = np.fft.fft(
        np.fft.fft(np.invert(nodwell).astype(int), 2 * m - 1, axis=0),
        2 * n - 1,
        axis=1,
    )
    # [Step 2] Multiply the relevant transforms and invert to obtain the
    # equivalent convolutions
    rawCorr = np.fft.fftshift(
        np.real(np.fft.ifft(np.fft.ifft(Fx * np.conj(Fx), axis=1), axis=0)),
        axes=(0, 1),
    )
    sums_x = np.fft.fftshift(
        np.real(np.fft.ifft(np.fft.ifft(np.conj(Fx) * Fn, axis=1), axis=0)),
        axes=(0, 1),
    )
    sumOfSquares_x = np.fft.fftshift(
        np.real(np.fft.ifft(np.fft.ifft(Fn * np.conj(FsumOfSquares_x), axis=1), axis=0)),
        axes=(0, 1),
    )
    N = np.fft.fftshift(
        np.real(np.fft.ifft(np.fft.ifft(Fn * np.conj(Fn), axis=1), axis=0)),
        axes=(0, 1),
    )
    # [Step 3] Account for rounding errors.
    rawCorr[np.abs(rawCorr) < tol] = 0
    sums_x[np.abs(sums_x) < tol] = 0
    sumOfSquares_x[np.abs(sumOfSquares_x) < tol] = 0
    N = np.round(N)
    N[N <= 1] = np.nan
    # [Step 4] Compute correlation matrix
    mapStd = np.sqrt((sumOfSquares_x * N) - sums_x**2)
    mapCovar = (rawCorr * N) - sums_x * sums_x[::-1, :, :][:, ::-1, :][:, :, :]
    A = np.squeeze(mapCovar / mapStd / mapStd[::-1, :, :][:, ::-1, :][:, :, :])
    A = np.nan_to_num(np.asarray(A), -1)  # reshape and decorrelate NaN's
    return A


def circle_mask(autocorr, relative_radius=1, in_val=1.0, out_val=0.0):
    """Get a mask to crop out the outer radius of an autocorrelogram"""
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