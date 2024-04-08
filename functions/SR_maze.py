'''The following script is meant to contain functions which quickly generate successor representations from mazes.
Here we also include convenient plotting functions.
See the 'Stachenfeld_representations.ipynb' notebook for more details.''' 

## SETUP ##
import numpy as np
import pandas as pd
import networkx as nx

import plotting as mp
import representations as mr


def get_maze_SR_df(maze, discount_factor=0.95):
    '''This function will return a random walk SR matrix for a given maze.
        Note that only nodes will be considered states and edges represent valid transitions.
        Using a fine-maze structure is recommended.
        INPUT: a maze networkx object and discount factor (aka gamma)
        OUTPUT: pandas dataframe indexed according to node/edge label of maze'''
    
    #Random walk successor representation

    #allow for indexed mazes / otherwise treat as networkx object.
    if type(maze) == str:
        maze = mr.get_simple_maze(maze)
        maze = mr.get_extended_simple_maze(maze)
        
    #First we take an adjacency matrix and add the possibility of standing still (identity) to get all possible transitions:
    adjacency = nx.adjacency_matrix(maze).toarray() #store adjacency in an array
    norm_adjacency = adjacency/max(adjacency.flatten()) #normalise so that all adjacency values are 1.
    np.fill_diagonal(norm_adjacency,1) #has no output, but makes all diagonals equal to 1, so standing still is a valid transition

    #Then we want to implement random walk policy, making each possible transition equally likely
    for c in range(np.size(adjacency[:][0])): #for each column
        norm_adjacency[:,c] = norm_adjacency[:,c]/sum(norm_adjacency[:,c]) #make each column a probability over transitions. Remember we index by [row,column].

    T_matrix = norm_adjacency #Renaming after the above

    #Compute successor representation, by the inverse of I-gT
    temp_matrix =  np.identity(T_matrix.shape[1])-discount_factor*T_matrix #(I - gammaT)^-1, so gamma here is discount factor defined above.
    SR_matrix = np.linalg.inv(temp_matrix)
    
    SR_df = pd.DataFrame(SR_matrix)
    SR_df.index=nx.get_node_attributes(maze, 'label').values()
    
    return SR_df

def plot_SR_fields_undirected(maze_idx, SR_df,n_plots = 'all'):
    ''' This plots SR place fields onto 'simple maze's in 5x5 panels
        until all columns of the SR matrix are plotted
        INPUT: maze_idx and SR_df. This accepts an SR_df which includes edges or not.'''

    maze = mr.get_simple_maze(maze_idx)

    maze_fine = mr.get_extended_simple_maze(maze) #we need the finer-scale for plotting heatmaps in either case.
    label = nx.get_node_attributes(maze_fine, 'label').values()
    
    if len(SR_df)==len(label):
        maze = maze_fine
    
    mp.plt.figure(figsize=(10, 10), dpi=300)

    # Here we plot the columns in sets of 25

    if n_plots == 'all': #sets n_plots to all columns unless specified.
        n_plots = len(SR_df.columns)

    for each_set in range(int(np.ceil(n_plots/25))): #For each set of 25 columns of the SR_matrix / place cell we want to plot.
        fig, axs = mp.plt.subplots() #Make a big figure for each set of 25 plots
        axs.axis('off')
        for plot in range(25): #plot each in a 5x5 panel:
            plot_position = plot+1 #must be integer between 1 and 25.
            ax = fig.add_subplot(5,5,plot_position)
            ax.set_aspect('equal')
            ax.axis('off')

            value = SR_df[plot*(each_set*1)] #Extract the column which we want to plot. This is a multiple of 25 for each set.
            #if we're on the small maze, we want the colour value of edges to map to 0.
            if len(SR_df)<len(label):
                value = mp.np.pad(value, (0, maze.number_of_edges()), 'constant') #SR matrix only has values for each node, so now we put a 0 for each edge.

            values = mp.pd.Series(data=value, index=label)
            values.index.name='maze_position' #required for the plot function to work

            mp.plot_simple_heatmap(mr.get_simple_maze(maze_idx), #We need to map onto the 'small' maze.
                                values,
                                ax,
                                node_size = 10, #these dimensions work with a panel of 25
                                edge_size = 2, #dimension works with a panel of 25
                                value_label='',
                                title='')
                

