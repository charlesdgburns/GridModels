'''This is code to decode place from firingrates of spatially stable cells.

Some code here is stolen from RatInABox demo'''

## imports ##

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

## global variables ##

## functions ##
def train_decoder(train_nav_rates_df, subsample_n = 5, type = "GP"):
    """t_start and t_end allow you to pick the poritions of the saved data to train on.
    Parameters:
    -----------
    train_nav_rates_df: pd.DataFrame()
        multiindex with columns 'time', 'firingrate','centroid_position',
    subsample_n: int()
        takes every subsample_n-th row of the data to train decoder on,
    type: str 
        either "GP" for Gaussian Process or "LR" for Linear Regression.
    
    Returns:
    --------
    decoder: a sklearn model fit to the data.
    
    Note: 
    use output to decode using decoder.predict(test_df.firingrate.values)"""#
    
    # Get training data
    t = train_nav_rates_df.time.values
    t = t[::subsample_n]  # subsample data for training (most of it is redundant anyway)
    fr = train_nav_rates_df.firingrate.values[::subsample_n]
    pos = train_nav_rates_df.centroid_position.values[::subsample_n]
    # set up models
    if type == "GP":
        decoder = GaussianProcessRegressor(
            alpha=0.01,
            #we scale kernel size with typical input size ~sqrt(N)
            kernel=RBF(1*np.sqrt(len(train_nav_rates_df.firingrate.columns) / 20), 
                        length_scale_bounds="fixed"),
            )
    elif type == "LR":
        decoder = Ridge(alpha=0.01)
    # train decoding models
    decoder.fit(X=fr, y=pos)

    return decoder


def compute_decoding_error(decoding_model, test_nav_rates_df):
    """Take a navigation_rates_df and add decoded position columns
    Returns: decoding errors per position in cm"""
    # decode position from the data add to the dataframe
    decoded_pos = decoding_model.predict(test_nav_rates_df.firingrate.values)
    actual_pos = test_nav_rates_df.centroid_position.values
    decoding_errors_cm = np.linalg.norm(actual_pos-decoded_pos, axis = 1)*100
    return decoding_errors_cm