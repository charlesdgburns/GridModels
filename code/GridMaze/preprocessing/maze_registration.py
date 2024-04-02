"""This file registers video pixel coordinates of select towers on a maze for later use to correct for fish-eye distortion. A quality control step is included
to ensure that the camera does not move over the recording period."""
# %% imports
import os
import csv
import cv2
import numpy as np
import pandas as pd
import moviepy.editor as mpy
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from .get_sessions_data_directory import get_data_directory_df
from ..maze.representations import _get_node_positions_dict, get_simple_nodes_dict

# %% Global variables
ALIGNMENT_NODES = ["A1", "A4", "A7", "C3", "C5", "D1", "D4", "D7", "E3", "E5", "G1", "G4", "G7"]
RAW_VIDEO_PATH = "../data/raw_data/video"
SESSIONS_DATA_DIRECTORY_DF = get_data_directory_df()
MAZE_REGISTRATION_PATH = os.path.join(RAW_VIDEO_PATH, "maze_registration.tsv")


# %% Get tower center pixel corrdinates from video
def get_tower_center_coordinates_from_video(video_path, alignment_nodes) -> dict:
    """Returns a dict of tower_label: (x,y) pixel coordinates, for towers specified in alignment_nodes, from a video file."""
    video = mpy.VideoFileClip(video_path)
    temp_image_path = os.path.join(RAW_VIDEO_PATH, "temp_image.png")
    video.save_frame(temp_image_path, t=0)
    image = cv2.imread(temp_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    os.remove(temp_image_path)
    tower_label2pixel_coords = {}
    for tower_label in alignment_nodes:
        print(f"click on the center of {tower_label}")
        pixel_coords = get_pixel_coords_from_image(image, tower_label)
        tower_label2pixel_coords[tower_label] = pixel_coords[0]
    return tower_label2pixel_coords


def get_average_tower_coords(tower2pixel_coord_replicates: list) -> dict:
    tower_pixel_coords = {}
    for tower_label in tower2pixel_coord_replicates[0].keys():
        x = [i[tower_label][0] for i in tower2pixel_coord_replicates]
        y = [i[tower_label][1] for i in tower2pixel_coord_replicates]
        tower_pixel_coords[tower_label] = (np.mean(x), np.mean(y))
    return tower_pixel_coords


def get_pixel_coords_from_image(image, label):
    """Returns pixel coordinates of a click on an image"""
    plt.figure(figsize=(15, 10))
    orientation = [0, image.shape[1], 0, image.shape[0]]
    plt.imshow(image, extent=orientation)
    plt.title(f"click on the center of {label}")
    plt.tight_layout()
    plt.axis("off")
    plt.show()
    pixel_coords = plt.ginput(n=1, timeout=0, show_clicks=True)
    plt.close()
    return pixel_coords


# %% Quality control
def run_tower_alignment_quality_control(click_variance_threshold, data_directory_df, alignment_nodes):
    """Returns True if the camera alignment is consistent between the first and last session in the data_directory_df"""
    alignment_offset_dict = compare_tower_alignment_between_first_and_last_session(data_directory_df, alignment_nodes)
    if all([i < click_variance_threshold for i in alignment_offset_dict.values()]):
        print("camera alignment is consistent between first and last session")
        return True
    else:
        print("camera alignment is not consistent between first and last session")
        return False


def get_tower_click_variance(tower_coord_dicts):
    """Returns the standard deviation of the distance between the pixel coordinates of the same tower in a list of tower_coord_dicts"""
    n = len(tower_coord_dicts)
    permutations = [(i, j) for i in range(n) for j in range(n) if i < j]
    offset_distances = []
    for x, y in permutations:
        for alignment_node in [i for i in tower_coord_dicts[0].keys()]:
            offset_distance = euclidean(tower_coord_dicts[x][alignment_node], tower_coord_dicts[y][alignment_node])
            offset_distances.append(offset_distance)
    return np.std(offset_distances)


def compare_tower_alignment_between_first_and_last_session(data_directory_df, alignment_nodes):
    first_session_video_path = data_directory_df.iloc[np.argmin(data_directory_df["datetime"])]["video_path"]
    last_session_video_path = data_directory_df.iloc[np.argmax(data_directory_df["datetime"])]["video_path"]
    first_session_coordinates = get_tower_center_coordinates_from_video(first_session_video_path, alignment_nodes)
    last_session_coordinates = get_tower_center_coordinates_from_video(last_session_video_path, alignment_nodes)
    alignment_offset_dict = {}
    for alignment_node in alignment_nodes:
        offset_distance = euclidean(first_session_coordinates[alignment_node], last_session_coordinates[alignment_node])
        alignment_offset_dict[alignment_node] = offset_distance
    return alignment_offset_dict


# %% Save and load tower alignment coordinates
def save_tower_alignment_coords(raw_video_path, tower_label2pixel_coords):
    """Save tower alignment coordinates to a .tsv file in the raw video directory: .../tower_center_pixel_coords.tsv"""
    with open(os.path.join(raw_video_path, "maze_registration.tsv"), "w") as outfile:
        writer = csv.writer(outfile, delimiter="\t")
        writer.writerow(["tower_label", "x", "y"])
        for tower_label, pixel_coords in tower_label2pixel_coords.items():
            writer.writerow([tower_label, pixel_coords[0], pixel_coords[1]])
    return


def load_tower_pixel_coords(tower_alignment_coords_path) -> dict:
    """Loads tower alignment coordinates from tower_center_pixel_coords.tsv (raw video directory) as dict of tower_label: (x,y)"""
    tower_label2pixel_coords = {}
    with open(tower_alignment_coords_path, "r") as infile:
        reader = csv.reader(infile, delimiter="\t")
        for row in [i for i in reader][1:]:
            tower_label2pixel_coords[row[0]] = (float(row[1]), float(row[2]))
    return tower_label2pixel_coords


# %% Load calibration coordinates df
def get_calibration_coordinates_df():
    """Returns a dataframe with pixel and physical coordinates for each tower"""
    label2pixel_coords = load_tower_pixel_coords(MAZE_REGISTRATION_PATH)
    label2simple_node = get_simple_nodes_dict()
    simple_node2physical_coords = _get_node_positions_dict()
    calibration_node2pixel_coords = {
        label2simple_node[tower]: label2pixel_coords[tower] for tower in label2pixel_coords.keys()
    }
    calibration_node2physical_coords = {
        k: v for k, v in simple_node2physical_coords.items() if k in calibration_node2pixel_coords.keys()
    }
    pixel_physical_coords_df = pd.DataFrame(
        {
            "simple_node_label": label2pixel_coords.keys(),
            "calibration_node": calibration_node2pixel_coords.keys(),
            "physical_coords": calibration_node2physical_coords.values(),
            "pixel_coords": calibration_node2pixel_coords.values(),
        }
    )
    return pixel_physical_coords_df


def get_image_size_from_video():
    """Returns the image size of the videos in the raw_data directory, using 1st frame from the 1st video as an example image
    Output is (height, width)"""
    example_video_path = SESSIONS_DATA_DIRECTORY_DF.iloc[np.argmin(SESSIONS_DATA_DIRECTORY_DF["datetime"])][
        "video_path"
    ]  # eg, first session
    temp_image_path = os.path.join(RAW_VIDEO_PATH, "temp_image.png")
    video = mpy.VideoFileClip(example_video_path)
    video.save_frame(temp_image_path, t=0)
    image = cv2.imread(temp_image_path)
    image_size = image.shape[:2]
    os.remove(temp_image_path)
    return image_size


# %% Get experiment alignment coordinates
if __name__ == "__main__":  # don't run if module is imported
    # get preliminary tower pixel coords
    first_session_video_path = SESSIONS_DATA_DIRECTORY_DF.iloc[np.argmin(SESSIONS_DATA_DIRECTORY_DF["datetime"])][
        "video_path"
    ]  # use first session to define preliminary tower pixel coords
    tower2pixel_coord_replicates = []
    n_replicates = 3
    for i in range(n_replicates):  # get ready to click!
        tower2pixel_coord_replicates.append(
            get_tower_center_coordinates_from_video(first_session_video_path, ALIGNMENT_NODES)
        )
    preliminary_tower_pixel_coordinates = get_average_tower_coords(tower2pixel_coord_replicates)

    # quality control
    click_variance = get_tower_click_variance(tower2pixel_coord_replicates)
    click_variance_threshold = 5 * click_variance
    qc_pass = run_tower_alignment_quality_control(click_variance_threshold, SESSIONS_DATA_DIRECTORY_DF, ALIGNMENT_NODES)

    # save tower alignment coordinates if quality control passes
    if qc_pass:
        save_tower_alignment_coords(RAW_VIDEO_PATH, preliminary_tower_pixel_coordinates)
        print("tower alignment coordinates saved")


# %%
