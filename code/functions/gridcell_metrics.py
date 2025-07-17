'''This script is primarily to help identify and measure grid cells in populations of cells.'''

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Grid score calculations.

This is copied directly from the google deepmind repository 
https://github.com/google-deepmind/grid-cells/blob/master/scores.py

Combined with orientation and scale finding code from NeuralPlayground:
https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/neuralplayground/comparison/metrics.py 
"""

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import os
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage

#for Gardner 2022 grid cell classification
import umap 
from sklearn.cluster import DBSCAN

# local imports

from GridMaze.analysis.cluster_tuning import spatial

np.seterr(divide="ignore", invalid="ignore")  # there will be divide by zeros. Ignore these


def circle_mask(size, radius, in_val=1.0, out_val=0.0):
    """Calculating the grid scores with different radius.
    Size = [n_bins_x,n_bins_y],
    radius = int proportional to number of bins"""
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


def get_mask_parameters(
    ratemap,
    coords_range,
    max_outer_radius=1,
    n_radii=16,
    inner_radius_buffer_factor=1,
):
    """Estimates inner radius of the autocorrelogram,
    returns mask parameters for expanding outer radius.
    """
    n_bins = min(ratemap.shape)
    initial_mask = [(0, 1)]
    scorer = GridScorer(n_bins, coords_range, initial_mask)
    # sac = scorer.calculate_sac(ratemap)
    sac = spatial.autoCorr2D(np.nan_to_num(ratemap.copy(), 0), np.isnan(ratemap.copy()))
    mask_min = get_inner_radius(sac, inner_radius_buffer_factor)
    mask_parameters = [(mask_min, x) for x in np.linspace(mask_min + 0.1, max_outer_radius, num=n_radii)]
    return mask_parameters


def get_inner_radius(sac, inner_radius_buffer_factor=1):
    """Returns the number of bins in the inner radius"""
    midpoint = int(np.floor(len(sac) / 2))
    # we want to estimate the number of bins where the inner peak stops dropping
    inner_peak_bins_x = [sac[midpoint, midpoint]]
    inner_peak_bins_y = [sac[midpoint, midpoint]]
    for each_increment in range(midpoint):
        x_amplitude = sac[midpoint + each_increment, midpoint]
        y_amplitude = sac[midpoint, midpoint + each_increment]
        if x_amplitude < inner_peak_bins_x[-1]:
            inner_peak_bins_x.append(x_amplitude)

        if y_amplitude < inner_peak_bins_y[-1]:
            inner_peak_bins_y.append(y_amplitude)

        if (x_amplitude < inner_peak_bins_x[-1]) and (y_amplitude < inner_peak_bins_y[-1]):
            break
    mean_n_bins = (len(inner_peak_bins_x) + len(inner_peak_bins_y)) / 2
    mask_min = mean_n_bins * inner_radius_buffer_factor / len(sac)  # proportion of inner radius relative full size.
    return mask_min


class GridScorer(object):
    """Class for scoring ratemaps given trajectories."""

    def __init__(self, nbins, coords_range, mask_parameters, min_max=False):
        """Scoring ratemaps given trajectories.

        Args:
          nbins: Number of bins per dimension in the ratemap.
          coords_range: Environment coordinates range.
          mask_parameters: parameters for the masks that analyze the angular
            autocorrelation of the 2D autocorrelation.
          min_max: Correction.
        """
        self._nbins = nbins
        self._min_max = min_max
        self._coords_range = coords_range
        self._corr_angles = [30, 45, 60, 90, 120, 135, 150]
        # Create all masks
        self._masks = [
            (self._get_ring_mask(mask_min, mask_max), (mask_min, mask_max)) for mask_min, mask_max in mask_parameters
        ]
        # Mask for hiding the parts of the SAC that are never used
        self._plotting_sac_mask = circle_mask(
            [self._nbins * 2 - 1, self._nbins * 2 - 1], self._nbins, in_val=1.0, out_val=np.nan
        )

    def calculate_ratemap(self, xs, ys, activations, statistic="mean"):
        return scipy.stats.binned_statistic_2d(
            xs, ys, activations, bins=self._nbins, statistic=statistic, range=self._coords_range
        )[0]

    def _get_ring_mask(self, mask_min, mask_max):
        n_points = [self._nbins * 2 - 1, self._nbins * 2 - 1]
        return circle_mask(n_points, mask_max * self._nbins) * (1 - circle_mask(n_points, mask_min * self._nbins))

    def grid_scores(self, corr):
        grid_score_60_minmax = np.minimum(corr[60], corr[120]) - np.maximum(corr[30], np.maximum(corr[90], corr[150]))
        grid_score_60 = (corr[60] + corr[120]) / 2 - (corr[30] + corr[90] + corr[150]) / 3  #
        grid_score_90 = corr[90] - (corr[45] + corr[135]) / 2
        return grid_score_60_minmax, grid_score_60, grid_score_90

    def calculate_sac(self, seq1):
        """Calculating spatial autocorrelogram.'
        ##Deepmind method - doesn't work well on the Maze structures"""
        seq2 = seq1

        def filter2(b, x):
            stencil = np.rot90(b, 2)
            return scipy.signal.convolve2d(x, stencil, mode="full")

        seq1 = np.nan_to_num(seq1)
        seq2 = np.nan_to_num(seq2)

        ones_seq1 = np.ones(seq1.shape)
        ones_seq1[np.isnan(seq1)] = 0
        ones_seq2 = np.ones(seq2.shape)
        ones_seq2[np.isnan(seq2)] = 0

        seq1[np.isnan(seq1)] = 0
        seq2[np.isnan(seq2)] = 0

        seq1_sq = np.square(seq1)
        seq2_sq = np.square(seq2)

        seq1_x_seq2 = filter2(seq1, seq2)
        sum_seq1 = filter2(seq1, ones_seq2)
        sum_seq2 = filter2(ones_seq1, seq2)
        sum_seq1_sq = filter2(seq1_sq, ones_seq2)
        sum_seq2_sq = filter2(ones_seq1, seq2_sq)
        n_bins = filter2(ones_seq1, ones_seq2)
        n_bins_sq = np.square(n_bins)

        std_seq1 = np.power(
            np.subtract(np.divide(sum_seq1_sq, n_bins), (np.divide(np.square(sum_seq1), n_bins_sq))), 0.5
        )
        std_seq2 = np.power(
            np.subtract(np.divide(sum_seq2_sq, n_bins), (np.divide(np.square(sum_seq2), n_bins_sq))), 0.5
        )
        covar = np.subtract(np.divide(seq1_x_seq2, n_bins), np.divide(np.multiply(sum_seq1, sum_seq2), n_bins_sq))
        x_coef = np.divide(covar, np.multiply(std_seq1, std_seq2))
        x_coef = np.real(x_coef)
        x_coef = np.nan_to_num(x_coef)
        return x_coef

    def rotated_sacs(self, sac, angles):
        return [scipy.ndimage.interpolation.rotate(sac, angle, reshape=False) for angle in angles]

    def get_grid_scores_for_mask(self, sac, rotated_sacs, mask):
        """Calculate Pearson correlations of area inside mask at corr_angles."""
        masked_sac = sac * mask
        ring_area = np.sum(mask)
        # Calculate dc on the ring area
        masked_sac_mean = np.sum(masked_sac) / ring_area
        # Center the sac values inside the ring
        masked_sac_centered = (masked_sac - masked_sac_mean) * mask
        variance = np.sum(masked_sac_centered**2) / ring_area + 1e-5
        corrs = dict()
        for angle, rotated_sac in zip(self._corr_angles, rotated_sacs):
            masked_rotated_sac = (rotated_sac - masked_sac_mean) * mask
            cross_prod = np.sum(masked_sac_centered * masked_rotated_sac) / ring_area
            corrs[angle] = cross_prod / variance
        grid_score_60_minmax, grid_score_60, grid_score_90 = self.grid_scores(corrs)
        return grid_score_60_minmax, grid_score_60, grid_score_90, variance

    def get_grid_properties(self, rate_map):
        """Get summary of scores for grid cells and estimated orientation and scale.

        OUTPUT: a dictionary containing
        'gridscore_60_minmax': gridscore computed using min and max
        'gridscore_60': gridscore computed with averages
        'gridscore_90': gridscore computed with averages
        'max_60_mask': (inner_radius, outer_radius) tuple for generating the mask used to compute maximal 90 scores
        'max_90_mask': (inner_radius, outer_radius) tuple for generating mask used to compute maximal 90 scores
        'scale_60': the gridscale estimated with max_60_mask
        'scale_90': gridscale estimated with max_90_mask
        'orientation_60': grid orientation estimated with max_60_mask
        'orientation_90': grid orientation estimated with max_90_mask
        'gridpeaks_60': [[x,y],[x,y],...] peak coordinates for hexagonal orientation and scale
        'gridpeaks_90': [[x,y],[x,y],...] peak coordinates for square orientation and scale
        """

        # sac = self.calculate_sac(rate_map)
        sac = spatial.autoCorr2D(np.nan_to_num(rate_map.copy(), 0), np.isnan(rate_map.copy()))
        rotated_sacs = self.rotated_sacs(sac, self._corr_angles)

        scores = [
            self.get_grid_scores_for_mask(sac, rotated_sacs, mask)
            for mask, mask_params in self._masks  # pylint: disable=unused-variable
        ]
        scores_60_minmax, scores_60, scores_90, variances = map(
            np.asarray, zip(*scores)
        )  # pylint: disable=unused-variable
        max_60_ind = np.argmax(scores_60)
        max_90_ind = np.argmax(scores_90)
        max_60_mask = self._masks[max_60_ind][1]
        max_90_mask = self._masks[max_90_ind][1]
        orientation_60, scale_60, gridpeaks_60 = self.get_orientation_scale(sac, max_60_mask)
        orientation_90, scale_90, gridpeaks_90 = self.get_orientation_scale(sac, max_90_mask)

        dict = {
            "sac": sac,
            "gridscore_60_minmax": scores_60_minmax[max_60_ind],
            "gridscore_60": scores_60[max_60_ind],
            "gridscore_90": scores_90[max_90_ind],
            "max_60_mask": max_60_mask,
            "max_90_mask": max_90_mask,
            "orientation_60": orientation_60,
            "orientation_90": orientation_90,
            "scale_60": scale_60,
            "scale_90": scale_90,
            "gridpeaks_60": gridpeaks_60,
            "gridpeaks_90": gridpeaks_90,
        }

        return dict

    def get_orientation_scale(self, sac, max_score_mask, deg=True, return_coords=True):
        """
        INPUT: spatial autocorrelogram and mask parameters for the maximal score (60deg or 90deg mask)
        OUTPUT: estimated scale and orientation of the grid cell.
        Orientation: counter-clockwise angle to the nearest peak from 3-o'clock#
        Scale: median distance to (non-central) peaks on spatial autocorrelogram
        """
        # here we want to identify peaks within an anulus
        mask = self._get_ring_mask(max_score_mask[0], max_score_mask[1])
        mask[mask == 0] = np.nan
        inner_radius = get_inner_radius(sac)
        min_distance = int(
            inner_radius * len(sac) / 4
        )  # minimum distance between peaks, derived from inner radius of mask
        if min_distance < 0:
            print(inner_radius)
            print(len(sac))
        peak_coords = skimage.feature.peak_local_max(np.nan_to_num(mask * sac, nan=-1), min_distance, threshold_rel=0.5)
        centre = np.floor(np.array(np.shape(sac.copy())) / 2)
        centred_peak_coords = peak_coords - centre

        # Skip cases where too few peaks are identified
        if len(centred_peak_coords) < 3:
            scale = np.nan
            orientation = np.nan

        else:
            # we compute scale as the median distance to peak, then rescale it
            peak_dist_to_centre = np.hypot(centred_peak_coords.T[0], centred_peak_coords.T[1])
            scale_in_sac_bins = np.median(peak_dist_to_centre)  # here given in bins of sac
            meters_per_rm_bin = np.diff(np.mean(self._coords_range, axis=0)) / self._nbins
            rm_bins_per_sac_bin = (2 * self._nbins - 1) / (
                2 * self._nbins
            )  # close to 1, since sac barely scales bin distances.
            scale = (scale_in_sac_bins * rm_bins_per_sac_bin * meters_per_rm_bin)[
                0
            ]  # give scale in unit of original ratemap coordinates
            # get polar angle
            x = centred_peak_coords.T[1]
            y = centred_peak_coords.T[0]
            if deg:
                theta = (
                    180.0 * -np.arctan2(y, x) / np.pi
                )  # note the - sign here to compute angle counterclockwise from 3pm
                orientation = np.sort(theta.compress(theta >= 0))[0]
            else:
                theta = -np.arctan2(y, x)  # note the - sign here to compute angle counterclockwise from 3pm
                orientation = np.sort(theta.compress(theta >= 0))[0]

        # lastly, return the desired value
        if return_coords:
            return (
                orientation,
                scale,
                peak_coords,
            )  # NB: coords are returned in order y,x ... this is weird but live with it
        else:
            return orientation, scale

    def plot_sac(
        self, sac, mask_params=None, peak_coords=None, ax=None, title=None, *args, **kwargs
    ):  # pylint: disable=keyword-arg-before-vararg
        """Plot spatial autocorrelogram -> from ratemap."""
        if ax is None:
            ax = plt.gca()
        # Plot the sac
        ax.imshow(sac, interpolation="none", *args, **kwargs)
        # ax.pcolormesh(useful_sac, *args, **kwargs)
        # Plot a ring for the adequate mask
        if mask_params is not None:
            center = self._nbins - 1
            ax.add_artist(
                plt.Circle(
                    (center, center),
                    mask_params[0] * self._nbins,
                    # lw=bump_size,
                    fill=False,
                    edgecolor="k",
                )
            )
            ax.add_artist(
                plt.Circle(
                    (center, center),
                    mask_params[1] * self._nbins,
                    # lw=bump_size,
                    fill=False,
                    edgecolor="k",
                )
            )
            ax.axis("off")
        if peak_coords is not None:
            ax.scatter(peak_coords.T[1], peak_coords.T[0], color="black")
        if title is not None:
            ax.set_title(title)


# %% UMAP projection classification
# Here we write code to perform UMAP classification of grid cells, rather than shuffled null distributions
from pathlib import Path
from GridMaze.analysis.core import load_data
from GridMaze.analysis.core import filter
import pandas as pd
import numpy as np


'''
The following function is useful for interactive sessions

from GridMaze2.analysis.core import get_clusters as gc

def plot_examples_from_umap_group(umap_df,labels):
  plot_df = pd.DataFrame({'CUID':umap_df.index,'labels':labels})
  for each_label in np.unique(labels):
      label_clusters = plot_df.query(f'labels=={each_label}')['CUID'].to_list()
      fig, ax = plt.subplots(2,3, figsize = (15,10))
      fig.suptitle(f'UMAP label {each_label} with {len(label_clusters)} clusters')
      counter = 0
      for each_cluster in label_clusters:
          axes = ax.flatten()[counter]
          try:
              ratemap,_,_ = gc.get_cluster(each_cluster).get_tuning_data('ratemap')
              axes.imshow(ratemap)
              axes.axis('off')
              counter += 1
          except:
              continue
          if counter == 6:
              break
  return None
'''

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