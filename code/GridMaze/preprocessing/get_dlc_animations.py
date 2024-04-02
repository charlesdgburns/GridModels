#%% 
"""UNFINISHED: This module generates animations of the DeepLabCut tracking data for quality control """
#%%% Imports
import networkx as nx
import pylab as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from . import get_frames_dfs as gt

#%%
TEST_DLC_PATH = '../data/raw_data/DeepLabCut/m2_2022-06-23-125203DLC_resnet50_PFC_BigMaze_GoalsSep28shuffle1_1030000.csv'
TEST_DLC_PATH2 = '../data/raw_data/DeepLabCut/m8_2022-07-29-144059DLC_resnet50_PFC_BigMaze_GoalsSep28shuffle1_1030000.csv'

#%%

#make a a graph representation of the mosue parts with networkx
def get_mouse_graph():
    node2bodypart_label = {0:'head_front',
                        1:'head_mid',
                        2:'head_back',
                        3:'ear_L',
                        4:'ear_R',
                        5:'body_front',
                        6:'body_mid',
                        7:'body_back'}
    mouse_graph_edges = [(0,1),(0,3),(0,4),(1,2),(2,3),(2,4),
                        (2,5),(3,4),(3,7),(4,7),(5,6),(6,7)]
    node2color = {0: 'red',
                1: 'orangered',
                2: 'yellow',
                3: 'chartreuse',
                4: 'cyan',
                5: 'dodgerblue',
                6: 'blue',
                7: 'blueviolet'}
    mouse_graph = nx.Graph()
    mouse_graph.add_nodes_from(node2bodypart_label.keys())
    mouse_graph.add_edges_from(mouse_graph_edges)
    nx.set_node_attributes(mouse_graph, node2bodypart_label, 'label')
    nx.set_node_attributes(mouse_graph, node2color, 'color')
    return mouse_graph


def draw_mouse_graph(positions):
    mouse_graph = get_mouse_graph()
    node2position = {node:tuple(pos) for node,pos in enumerate(positions)}
    if np.isnan(positions).any():
        nan_nodes = np.argwhere(np.isnan(positions).any(axis=1))[0]
        for i in nan_nodes: 
            mouse_graph.remove_node(i)
    plt.figure(1, clear=True, figsize=(5,5))
    nx.draw(mouse_graph,
            labels = nx.get_node_attributes(mouse_graph,'label'),
            node_color = list(nx.get_node_attributes(mouse_graph,'color').values()),
            pos = node2position)

def update_mouse_graph(frame, frame_coords):
    positions = frame_coords[frame]
    draw_mouse_graph(positions)

#%%
def save_mouse_graph_animation(file_path):
    dlc_df = gt.get_cleaned_dlc_df(file_path)
    no_bodyparts = len(dlc_df.columns.get_level_values(0).unique())
    no_frames = len(dlc_df)
    frame_coords = dlc_df.to_numpy().reshape((no_frames,no_bodyparts,2))
    fig = plt.figure(1, clear=True, figsize=(5,5))
    animation = FuncAnimation(fig, update_mouse_graph, frames=no_frames, fargs=(frame_coords, ), interval=200)
    animation.save('../results/mouse_graph_animation.mp4', fps=60)



# %% make movie of centroid moving over time 
def save_mouse_centroid_animation(file_path):
    dlc_df = gt.get_cleaned_dlc_df(file_path)
    no_bodyparts = len(dlc_df.columns.get_level_values(0).unique())
    no_frames = len(dlc_df)
    frame_coords = dlc_df.to_numpy().reshape((no_frames,no_bodyparts,2))
    centroid_positions = np.array([gt.get_centroid_position(position) for position in frame_coords])
    fig = plt.figure(1, clear=True, figsize=(5,5))
    animation = FuncAnimation(fig, update_centroid_plot, frames=no_frames, fargs=(centroid_positions, ), interval=200)
    animation.save('../results/centroid_animation.mp4', fps=60)

def plot_centroid(centroid):
    plt.figure(1, clear=True, figsize=(5,5))
    plt.plot(*centroid, '.k', markersize=10)
    plt.xlim(0, 1.47)
    plt.ylim(0, 1.47)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

def update_centroid_plot(frame, centroid_positions):
    centroid = centroid_positions[frame]
    plot_centroid(centroid)

   #%% 
   
def create_arrow(ax, x, y, dx, dy, width, color):
    arrow = FancyArrowPatch((x, y), (x + dx, y + dy), arrowstyle='->', mutation_scale=width*50, color=color)
    ax.add_patch(arrow)
    return arrow

def create_dot(ax, x, y, size, color):
    dot, = ax.plot(x, y, marker='o', markersize=size, color=color)
    return dot

def update_animation(frame, pos_dir, arrow, dot):
    x, y, t = pos_dir[frame]
    update_point_arrow_representation(x, y, t, arrow, dot)

def update_point_arrow_representation(x, y, theta, arrow, dot, arrow_length=0.15, arrow_width=0.1):
    theta_rad = np.radians(theta)
    dx = arrow_length * np.sin(theta_rad)
    dy = arrow_length * np.cos(theta_rad)
    start_x = x - dx / 2
    start_y = y - dy / 2

    # Update arrow
    arrow.set_positions((start_x, start_y), (start_x + dx, start_y + dy))

    # Update dot
    dot.set_xdata(x)
    dot.set_ydata(y)

def save_point_arrow_animation(session_directory):
    frames_df = gt.get_trajectories_df(session_directory)
    head_direction = gaussian_filter(frames_df['head_direction', 'values'], sigma=10)
    x = gaussian_filter(frames_df['centroid_position', 'x'], sigma=10)
    y = gaussian_filter(frames_df['centroid_position', 'y'], sigma=10)
    pos_dir = np.array([x,y,head_direction]).T
    no_frames = len(frames_df)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1.47)
    ax.set_ylim(0, 1.47)
    
    x, y, t = pos_dir[0]
    theta_rad = np.radians(t)
    dx = 0.15 * np.sin(theta_rad)
    dy = 0.15 * np.cos(theta_rad)
    start_x = x - dx / 2
    start_y = y - dy / 2

    arrow = create_arrow(ax, start_x, start_y, dx, dy, width=0.5, color='r')
    dot = create_dot(ax, x, y, size=10, color='grey')

    animation = FuncAnimation(fig, update_animation, frames=len(pos_dir), fargs=(pos_dir, arrow, dot), interval=200)
    animation.save('../results/point_arrow_animation2.mp4', fps=60)