

'''
Code to do Gardner et al. (2022) style UMAP projections of grid cells (clustering periodicity of spatial tuning patterns)
'''

# %% UMAP projection classification
# Here we write code to perform UMAP classification of grid cells, rather than shuffled null distributions
import os
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import umap 
from sklearn.cluster import DBSCAN
from GridModels.code.measures import spatial
from GridModels.code.measures.grid_scorer import GridScorer

def get_umap_cluster_labels(umap_df, eps=0.5, min_samples=10):
  reducer = umap.UMAP(random_state=42, 
                      metric = 'manhattan',# as in Gardner et al. 2022
                      n_neighbors=5, #same as Gardner et al. 2022
                      min_dist=0.05) #same as Gardner et al. 2022
  data_umap = reducer.fit_transform(umap_df)
  dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # Adjust `eps` as needed
  labels = dbscan.fit_predict(data_umap)
  plt.scatter(data_umap[:,0], data_umap[:,1], c=labels, alpha=0.5)
  plt.show()
  return labels


def get_umap_sacs_df(navigation_activity_df):
    """
    Variables:
    navigation_activity_df: pd.DataFrame
      with 'spike_count' or 'firing_rate' column and centroid_position column.,
    
    Returns:
    -------
    umap_df: pd.DataFrame() #(n_bins, n_clusters)
        an array of cropped spatial autocorrelograms flattened and z-scored.
    """
    # Get ratemaps and sac's for each cluster
    activity_column = [x for x in navigation_activity_df.columns for y in ['count','rate'] if y in x ]
    vector_list = []  # this will be rows
    cluster_unique_IDs = navigation_activity_df[activity_column].columns  # this will be columns
    pos = navigation_activity_df.centroid_position.to_numpy()
    for i, each_cluster in enumerate(cluster_unique_IDs):
        spikes = navigation_activity_df[activity_column][each_cluster].to_numpy().reshape(-1)
        if 'open_field' in each_cluster:
            ratemap, binx, biny = spatial.get_2D_ratemap(spikes,pos, x_size = 0.05, y_size =0.05, smooth_SD = 0,
                                                x_range=(0.2,1.2), y_range=(0.2,1.2))
        else:
            ratemap, binx, biny = spatial.get_2D_ratemap(spikes,pos, x_size = 0.05, y_size =0.05, smooth_SD = 0,
                                                  x_range=(0.05,1.35), y_range=(0.05,1.35))
    
        coords_range = [[min(binx),max(binx)],[min(biny),max(biny)]]
        sac = spatial.autoCorr2D(np.nan_to_num(ratemap.copy(), 0), np.isnan(ratemap.copy()))
        mask_parameters = [(0.3,1)]
        scorer = GridScorer(len(binx)-1, coords_range, mask_parameters)
        mask = scorer._get_ring_mask(mask_parameters[0][0],mask_parameters[0][1])
        vectorised = sac.flatten()[mask.flatten()==1] # exclude outside disc and vectorise
        z_scored =  (vectorised - vectorised.mean())/vectorised.std()
        vector_list.append(vectorised)
    sac_df = pd.DataFrame(dict(zip(cluster_unique_IDs, vector_list)))
    umap_df = sac_df.dropna(axis=1)
    umap_df = umap_df.T #transpose so that index is CUID and each column is the z-scored vectorised autocorrelogram.
    return umap_df  # (n_clusters,n_bins)


# %%