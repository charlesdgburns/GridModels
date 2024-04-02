"""This module populates processes pycontrol, video and raw ephys data into processed data in the processed data folder"""
# %% Imports
import os
import numpy as np
import pandas as pd
from . import get_frames_dfs as fd
from .get_session_info import save_session_info
from .get_frames_dfs import get_trajectories_df
from .convert_to_pycontrol_time import get_spike_pytimes
from .get_pycontrol_dfs import get_events_df, get_trials_df
from .get_sessions_data_directory import get_data_directory_df

# %% Global Variables
SESSIONS_DATA_DIRECTORY = get_data_directory_df()
PROCESSED_DATA_PATH = '../data/processed_data'
ANALYSIS_DATA_PATH = '../data/analysis_data'

# %% add processed pycontrol data to processed data folders
def populate_processed_pycontrol_data():
    """"""
    for session in SESSIONS_DATA_DIRECTORY.itertuples():
        processed_data_folder = os.path.join(PROCESSED_DATA_PATH,session.subject_ID,session.datetime_string)
        os.makedirs(processed_data_folder, exist_ok=True)
        # process pycontrol data
        events_df = get_events_df(session.pycontrol_path)
        events_df.to_csv(os.path.join(processed_data_folder, 'events.htsv'), index=False, sep='\t')
        trials_df = get_trials_df(session.pycontrol_path)
        #flatten multiindex columns to save as .htsv
        new_columns = ['trial', 'goal', 'errors']
        new_columns.extend([f'{x[0]}.{x[1]}' for x in trials_df.columns.to_flat_index() if x[0] not in ['trial', 'goal', 'errors']])
        trials_df.columns = new_columns
        trials_df.to_csv(os.path.join(processed_data_folder, 'trials.htsv'), index=False, sep='\t')
        save_session_info(session.pycontrol_path, processed_data_folder)

def populate_processed_frames_data():
    """"""
    for session_dir in SESSIONS_DATA_DIRECTORY.itertuples():
        processed_data_folder = os.path.join(PROCESSED_DATA_PATH,session_dir.subject_ID,session_dir.datetime_string)
        print(f'processing session: , {session_dir.subject_ID}, {session_dir.datetime_string}')
        os.makedirs(processed_data_folder, exist_ok=True)
        tracking_df = fd.get_tracking_df(session_dir)
        trajectories_df = fd.get_trajectories_df(session_dir)
        trial_info_df = fd.get_trial_info_df(session_dir)
        #flatten multiindex columns to save as .htsv
        tracking_df.columns = get_flattered_multiindex_columns(tracking_df)
        trajectories_df.columns = get_flattered_multiindex_columns(trajectories_df)
        tracking_df.to_csv(os.path.join(processed_data_folder, 'frames.tracking.htsv'), index=False, sep='\t')
        trajectories_df.to_csv(os.path.join(processed_data_folder, 'frames.trajectories.htsv'), index=False, sep='\t')
        trial_info_df.to_csv(os.path.join(processed_data_folder, 'frames.trialInfo.htsv'), index=False, sep='\t')

def populate_processed_ephys_data():
    """"""
    for session_dir in SESSIONS_DATA_DIRECTORY.itertuples():
        if any([isinstance(s_path, float) for s_path in session_dir]):
            continue  # Raw data files are missing (entry = float('nan'))
        print(f'Processing session: {session_dir.subject_ID} {session_dir.datetime_string}')
        # populate spike_pytimes.npy and spike_clusters.npy files
        processed_data_folder = os.path.join(PROCESSED_DATA_PATH,session_dir.subject_ID,session_dir.datetime_string)
        spike_pytimes = get_spike_pytimes(session_dir)
        not_nan = ~np.isnan(spike_pytimes) #spikes before & after pycontrol session
        spike_clusters = np.load(os.path.join(session_dir.phy_path, 'spike_clusters.npy'))
        spike_clusters = spike_clusters[not_nan]
        spike_pytimes = spike_pytimes[not_nan]
        cluster_metrics = pd.read_csv(os.path.join(session_dir.phy_path, 'cluster_KSLabel.tsv'), sep='\t')
        np.save(os.path.join(processed_data_folder, 'spike.clusters.npy'), spike_clusters)
        np.save(os.path.join(processed_data_folder, 'spike.times.npy'), spike_pytimes)
        cluster_metrics.to_csv(os.path.join(processed_data_folder, 'cluster.metrics.htsv'), sep='\t', index=False)

#%% Subfunctions
def get_flattered_multiindex_columns(df):
    """Returns a list of flat column names (str) where columns that were previously multiindex become level0_name.level1_name
    and single index columns stay level0_name"""
    return [f'{x[0]}.{x[1]}' if x[1] != '' else x[0] for x in df.columns.to_flat_index()]

def change_filename(current_filename, new_filename):
    """Changes the filename of a file in the processed data folder"""
    for subject in [x for x in os.listdir(PROCESSED_DATA_PATH) if not x.startswith('.')]:
        for session in [y for y in os.listdir(os.path.join(PROCESSED_DATA_PATH, subject)) if not y.startswith('.')]:
            session_path = os.path.join(PROCESSED_DATA_PATH, subject, session)
            if current_filename in os.listdir(session_path):
                current_filepath = os.path.join(session_path, current_filename)
                new_filepath = os.path.join(session_path, new_filename)
                os.rename(current_filepath, new_filepath)
            else:
                print(f'File {current_filename} not found in {session_path}')

def delete_file(filename):
    """Deletes all instances of a filname from processed data"""
    for subject in [x for x in os.listdir(PROCESSED_DATA_PATH) if not x.startswith('.')]:
        for session in [y for y in os.listdir(os.path.join(PROCESSED_DATA_PATH, subject)) if not y.startswith('.')]:
            session_path = os.path.join(PROCESSED_DATA_PATH, subject, session)
            if filename in os.listdir(session_path):
                filepath = os.path.join(session_path, filename)
                os.remove(filepath)
            else:
                print(f'File {filename} not found in {session_path}')


#%%
if __name__ == '__main__':
    populate_processed_pycontrol_data()
    populate_processed_ephys_data()
    populate_processed_frames_data()



