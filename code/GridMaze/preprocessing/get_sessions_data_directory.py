"""Build dataframe with all data filepaths associated with individual sessions"""

# %% Imports
import os
import numpy as np
import pandas as pd
import datetime as dt
from . import get_session_info as si

# %% Global variables
RAW_DATA_PATH = "../data/raw_data"
BOMBCELL_PATH = "../data/raw_data/bombcell"
KILOSORT_PATH = "../data/raw_data/kilosort"
PYCONTROL_PATH = "../data/raw_data/pycontrol"
EPHYS_PATH = "../data/raw_data/ephys"
VIDEO_PATH = "../data/raw_data/video"
SLEAP_PATH = "../data/raw_data/SLEAP"

# Manually specify sessions (pycontrol) name to be removed from analysis
REMOVE_SESSIONS = [
##
]  # restarted pycontrol


# %% Main function
def get_data_directory_df(filter_sessions=True):
    """"""
    base_directory_df = get_base_directory_df(PYCONTROL_PATH)
    directory_df = base_directory_df.copy()
    directory_df["ephys_timestamps_path"] = _get_ephys_timestamp_paths(base_directory_df, EPHYS_PATH)
    kilosort_foldernames, phy_paths = _get_phy_paths(base_directory_df, KILOSORT_PATH)
    directory_df["kilosort_foldername"] = kilosort_foldernames
    directory_df["phy_path"] = phy_paths
    directory_df["video_path"] = _get_video_paths(base_directory_df, VIDEO_PATH)
    directory_df["video_pinstate_path"] = _get_video_pinstate_paths(base_directory_df, VIDEO_PATH)
    directory_df["sleap_path"] = _get_sleap_paths(base_directory_df, SLEAP_PATH)
    if filter_sessions:  # remove duplicate or invalid sessions
        remove_sessions_mask = directory_df.pycontrol_path.apply(lambda x: os.path.split(x)[-1][:-4]).isin(
            REMOVE_SESSIONS
        )
        directory_df = directory_df[~remove_sessions_mask]
    return directory_df.sort_values(["subject_ID", "datetime"])


# %% Subfunctions
def get_base_directory_df(PYCONTROL_PATH):
    """start building sessions df from list of subject IDs and session datetimes. Defined from pycontrol raw data"""
    pycontrol_filenames = [py for py in os.listdir(PYCONTROL_PATH) if py[-4:] == ".tsv"]
    subject_IDs = [f.split(".")[0].split("-", 1)[0] for f in pycontrol_filenames]
    datetime_strings = [f.split(".")[0].split("-", 1)[1] for f in pycontrol_filenames]
    datetimes = [dt.datetime.strptime(datetime_string, "%Y-%m-%d-%H%M%S") for datetime_string in datetime_strings]
    iso_datetimes = [datetime.isoformat() for datetime in datetimes]
    pycontrol_paths = [os.path.join(PYCONTROL_PATH, f) for f in pycontrol_filenames]
    experimental_days = []
    maze_nos = []
    days_on_maze = []
    maze_structures = []
    goal_subsets = []
    goals = []
    for pycontrol_path in pycontrol_paths:
        session_info = si.get_session_info(pycontrol_path)
        experimental_days.append(session_info["experimental_day"])
        maze_nos.append(session_info["maze_number"])
        days_on_maze.append(session_info["day_on_maze"])
        maze_structures.append(session_info["maze_structure"])
        goal_subsets.append(session_info["goal_subset"])
        goals.append(session_info["goals"])
    base_sessions_df = pd.DataFrame(
        {
            "subject_ID": subject_IDs,
            "datetime": datetimes,
            "datetime_string": datetime_strings,
            "iso_datetime": iso_datetimes,
            "experimental_day": experimental_days,
            "maze_number": maze_nos,
            "day_on_maze": days_on_maze,
            "maze_structure": maze_structures,
            "goal_subset": goal_subsets,
            "goals": goals,
            "pycontrol_path": pycontrol_paths,
        }
    )
    return base_sessions_df


def _get_ephys_timestamp_paths(base_directory_df, ephys_path):
    """"""
    timestamp_paths = []  # ordered by row base_directory_df
    internal_timestamps_path = "experiment1/recording1/events/Rhythm_FPGA-109.0/TTL_1/timestamps.npy"
    for row in base_directory_df.itertuples():
        subject_ephys_sessions = os.listdir(os.path.join(ephys_path, row.subject_ID))
        subject_ephys_sessions = [
            ephys for ephys in subject_ephys_sessions if (ephys[0] != ".") and (ephys[-4:] != ".txt")
        ]
        subject_session_datetimes = [
            dt.datetime.strptime(ephys, "%Y-%m-%d_%H-%M-%S") for ephys in subject_ephys_sessions
        ]
        nearest_session = nearest_datetime(subject_session_datetimes, row.datetime)
        # quality control: if file datetime >5 mins from pycontrol datetime, not same session
        if abs(nearest_session - row.datetime).total_seconds() > 5 * 60:
            base_directory_df.loc[row.Index, "timestamps_path"] = np.nan
        ephys_folder = nearest_session.strftime("%Y-%m-%d_%H-%M-%S")
        base_directory_df.loc[row.Index, "ephys_foldername"] = ephys_folder
        # adjust filepaths based on which Record Node was used for recording (open ephys parameter)
        if "Record Node 121" in os.listdir(os.path.join(ephys_path, row.subject_ID, ephys_folder)):
            record_node = "Record Node 121"
        else:
            record_node = "Record Node 122"
        timestamp_paths.append(
            os.path.join(ephys_path, row.subject_ID, ephys_folder, record_node, internal_timestamps_path)
        )
    return timestamp_paths


def _get_phy_paths(base_directory_df, kilosort_path):
    kilosort_folder_names = []
    phy_paths = []
    for row in base_directory_df.itertuples():
        subject_ks_sessions = os.listdir(os.path.join(kilosort_path, row.subject_ID))
        subject_ks_sessions = [ks for ks in subject_ks_sessions if (ks[0] != ".") and (ks != "ignore")]
        try:
            ks_folder = next(ks for ks in subject_ks_sessions if ks[:19] == row.ephys_foldername)
            kilosort_folder_names.append(ks_folder)
            phy_paths.append(os.path.join(kilosort_path, row.subject_ID, ks_folder, "Phy"))
        except StopIteration:  # No matching kilosort output found.
            kilosort_folder_names.append(np.nan)
            phy_paths.append(np.nan)
    return kilosort_folder_names, phy_paths


def _get_video_pinstate_paths(base_directory_df, video_path):
    video_pinstate_paths = []
    session_pinstate_filenames = [i for i in os.listdir(video_path) if i[-4:] == ".csv"]
    session_pinstate_datetimes = [
        dt.datetime.strptime(i.split("_")[-1][:-4], "%Y-%m-%d-%H%M%S") for i in session_pinstate_filenames
    ]
    for row in base_directory_df.itertuples():
        nearest_session = nearest_datetime(session_pinstate_datetimes, row.datetime)
        pinstate_filename = session_pinstate_filenames[session_pinstate_datetimes.index(nearest_session)]
        if not run_file_qc(row.datetime, nearest_session, row.subject_ID, pinstate_filename.split("_")[0]):
            video_pinstate_paths.append(np.nan)
            continue
        pinstate_filepath = os.path.join(video_path, pinstate_filename)
        video_pinstate_paths.append(pinstate_filepath)
    return video_pinstate_paths


def _get_video_paths(base_directory_df, video_path):
    video_paths = []
    session_video_filenames = [i for i in os.listdir(video_path) if i[-4:] == ".mp4"]
    session_video_datetimes = [
        dt.datetime.strptime(i.split("_")[-1][:-4], "%Y-%m-%d-%H%M%S") for i in session_video_filenames
    ]
    for row in base_directory_df.itertuples():
        nearest_session = nearest_datetime(session_video_datetimes, row.datetime)
        video_filename = session_video_filenames[session_video_datetimes.index(nearest_session)]
        if not run_file_qc(row.datetime, nearest_session, row.subject_ID, video_filename.split("_")[0]):
            video_paths.append(np.nan)
            continue
        video_filepath = os.path.join(video_path, video_filename)
        video_paths.append(video_filepath)
    return video_paths


def _get_sleap_paths(base_directory_df, sleap_path):
    """"""
    sleap_paths = []
    session_sleap_filenames = [i for i in os.listdir(sleap_path) if i[-4:] == ".csv"]
    session_sleap_datetimes = [
        dt.datetime.strptime(i.split("_")[1][:-3], "%Y-%m-%d-%H%M%S") for i in session_sleap_filenames
    ]
    for row in base_directory_df.itertuples():
        nearest_session = nearest_datetime(session_sleap_datetimes, row.datetime)
        SLEAP_filename = session_sleap_filenames[session_sleap_datetimes.index(nearest_session)]
        if not run_file_qc(row.datetime, nearest_session, row.subject_ID, SLEAP_filename.split("_")[0]):
            sleap_paths.append(np.nan)
            continue
        SLEAP_filepath = os.path.join(sleap_path, SLEAP_filename)
        sleap_paths.append(SLEAP_filepath)
    return sleap_paths


def nearest_datetime(all_times, ref_time):
    """Finds nearest datetime in all_times to ref_time"""
    return min(all_times, key=lambda x: abs(x - ref_time))


# %% QC functions
def run_file_qc(pycontrol_datetime, file_datetime, pycontrol_subject_ID, file_subject_ID):
    """Checks if pycontrol and other data file are from the same session"""
    is_valid = True
    # if file generated more than 5mins after pycontrol, not same session
    if abs(pycontrol_datetime - file_datetime).total_seconds() > 60 * 5:
        is_valid = False
    # if subjects IDs not the same, not same session
    if pycontrol_subject_ID != file_subject_ID:
        is_valid = False
    return is_valid


def find_mutli_session_days_qc():
    """"""
    sessions_data_directory = get_data_directory_df(filter_sessions=False)
    subject_IDs = sessions_data_directory.subject_ID.unique()
    multi_session_days_dfs = []
    for subject_ID in subject_IDs:
        subject_sessions = sessions_data_directory[sessions_data_directory.subject_ID == subject_ID]
        dates = subject_sessions.datetime.dt.date.unique()
        for date in dates:
            date_sessions = subject_sessions[subject_sessions.datetime.dt.date == date]
            if len(date_sessions) > 1:
                multi_session_days_dfs.append(date_sessions)
    return pd.concat(multi_session_days_dfs)


# %%
