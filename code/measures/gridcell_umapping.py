

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

def get_open_field_umap_df(subject_ID:str):
  '''Function to get a large dataframe with all spatial autocorrelograms from open field sessions
  NB: we may want to drop MUA and bad units.'''
  analysis_data_paths = [] #get all paths
  subject_folder = f'../data/analysis_data/{subject_ID}'
  for subfolders in os.listdir(subject_folder):
    if 'open_field' in subfolders:
      analysis_data_paths.append(Path(f'{subject_folder}')/f'{subfolders}')
  
  #get all the spatial autocorrelograms:
  df_all = pd.DataFrame() #initialise dataframe for all sessions' sac's
  for analysis_data_path in analysis_data_paths:
    session_df = get_session_umap_sacs(analysis_data_path)
    df_all = pd.concat([df_all,session_df], axis=1)
  umap_df = df_all.dropna(axis=1)
  umap_df = umap_df.T #transpose so that index is CUID and each column is the z-scored vectorised autocorrelogram.
  return umap_df
  

def get_session_umap_sacs(analysis_data_path):
    """INPUT: path for processed data For each cluster in a session,
              returns an array of cropped spatial autocorrelograms flattened and z-scored.
    #(n_bins, n_clusters)"""
    # 1. Load data
    try:
        navigation_df = load_data.load(analysis_data_path / "frames.navigation.parquet")
        spikes_df = load_data.load(analysis_data_path / "frames.spikeCounts.parquet")
    except FileNotFoundError:
        print(f"Failed to load data for {analysis_data_path}. Returning None")
        return None

    # 1.1 filter away spikes without movements
    spikes_df = spikes_df.reset_index(drop=True)  # this step is required for maze filtering.
    navigation_activity_df = pd.concat([navigation_df, spikes_df], axis=1)
    if "maze" in analysis_data_path.parts[-1]:
        navigation_activity_df = filter.filter_navigation_rates_df(navigation_activity_df, True, True, False)
    else:
        navigation_activity_df = navigation_activity_df[navigation_activity_df.time.ge(0)]
        navigation_activity_df = navigation_activity_df[navigation_activity_df.moving]

    # 2 Get ratemaps and sac's for each cluster
    vector_list = []  # this will be rows
    cluster_unique_IDs = navigation_activity_df.spike_count.columns  # this will be columns
    pos = navigation_activity_df.centroid_position.to_numpy()
    for i, each_cluster in enumerate(cluster_unique_IDs):
        spikes = navigation_activity_df.spike_count[each_cluster].to_numpy().reshape(-1)
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
    return sac_df  # (n_bins, n_clusters)


# %%