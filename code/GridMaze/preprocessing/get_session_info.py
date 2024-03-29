"""This module get session information from pycontrol files using details manually specified in experiment_info.json"""
# %% Imports
import os
import json
import datetime as dt
from . import pycontrol_data_import as di

# %% Global variables
with open("../data/experiment_info.json", "r") as infile:
    EXP_INFO = json.load(infile)


# %% Main functions
def get_session_info(pycontrol_path):
    """Returns a dictionary containing session information associated with an input pyControl sessiion .txt file"""
    session = di.Session(pycontrol_path)
    session_info = {
        "subject_ID": _get_subject_ID(session),
        "session_date": _get_session_datetime(session),
        "experimental_day": _get_experimental_day(session),
        "maze_number": _get_maze_number_day(session)[0],
        "maze_structure": _get_maze_structure(session),
        "day_on_maze": _get_maze_number_day(session)[1],
        "goal_subset": _get_goal_subset(session),
        "goals": _get_goals(session),
        "reward_size": _get_reward_size(session),
    }
    return session_info


def save_session_info(pycontrol_path, output_path):
    """Writes a .json file to a specified output_filepath containing session information associated with an
    input pyControl session .txt file:
    - subject_ID: mouse number code (eg, m7)
    - session_date: calendar date the session was conducted
    - experimental_day: nth day since the 1st experimental session
    - maze_no: maze configuration number
    - maze_structure: maze structure defined as a list of edges [(A1, A2), (A2, A3), ...]
    - day_on_maze: nth day the subject has been exposed to the particular maze structure
    - goal_subset
    - goals: active goals during the session (all, subset_1 or subset_2)
    - reward_size: the size (in uL) of reward obtained on each trial
    """
    session = di.Session(pycontrol_path)
    session_info = get_session_info(pycontrol_path)
    with open((os.path.join(output_path, "session_info.json")), "w") as outfile:
        outfile.write(json.dumps(session_info, indent=4))
    return


# %% sub functions
def _get_goals(session):
    """Gets the goals acvtive during the session from data stored in the experiment_info.json file"""
    maze_no_day = _get_maze_number_day(session)
    goal_subset = _get_goal_subset(session)
    goals = EXP_INFO["maze_config2info"][f"maze_{maze_no_day[0]}"]["goals"][goal_subset]
    return goals


def _get_goal_subset(session):
    maze_no_day = _get_maze_number_day(session)
    goal_subset = EXP_INFO["maze_day2goals"][f"maze_{maze_no_day[0]}"][f"{maze_no_day[1]}"]
    return goal_subset


def _get_subject_ID(session):
    """'Get subject ID from session info"""
    subject_ID = "m" + str(session.subject_ID)
    return subject_ID


def _get_session_datetime(session):
    """'Get subject ID from session info"""
    return session.datetime_string


def _get_experimental_day(session):
    """Returns the number of days since the start of the experiment
    - Note: this counts days between maze configurations"""
    start_experiment_date = _convert_string2datetime(EXP_INFO["maze_config2info"]["maze_1"]["start"])
    experimental_day = session.datetime - start_experiment_date
    return experimental_day.days + 1


def _get_maze_number_day(session):
    """Returns a which maze structure the session used(1st element),
    and the number of days since starting training on that maze configuration (2nd element)"""
    maze_info = EXP_INFO["maze_config2info"]
    for maze in EXP_INFO["maze_config2info"].keys():
        start_date = _convert_string2datetime(maze_info[maze]["start"])
        end_date = _convert_string2datetime(maze_info[maze]["end"])
        if start_date <= session.datetime <= end_date:
            maze_number = int(maze.split("_")[-1])
            day_on_maze = session.datetime - start_date
    return maze_number, day_on_maze.days + 1


def _get_reward_size(session):
    """Retrieves the reward size used during the session
    - Note: only compatiable in sessions where reward size is fixed"""
    reward_size2dur = EXP_INFO["reward_size2dur"]
    reward_dur2size = {v: k for k, v in reward_size2dur.items()}  # Flips dictionary
    reward_duration = int(session.task_variables[0].split()[-1])
    return reward_dur2size[reward_duration] if reward_duration in reward_dur2size.keys() else f"{reward_duration}ms"


def _get_maze_structure(session):
    """Retrieves the maze structure, represented as a list of edges (eg, 'A1-A2')
    from the experiment_info.json file"""
    maze_no = _get_maze_number_day(session)[0]
    return EXP_INFO["maze_config2info"][f"maze_{maze_no}"]["structure"]


def _convert_string2datetime(string):
    """Converts datetimes stored as strings in the.json file into datetime.datetime objects"""
    return dt.datetime.strptime(string, "%m/%d/%Y, %H:%M:%S")


# %%
