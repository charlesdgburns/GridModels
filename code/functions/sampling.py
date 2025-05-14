## A set of functions to sample points and trajectories from mazes ##

#Setup

#Importing files from Peter Doohan's maze codebase:
import os
os.chdir('/content/drive/MyDrive/Colab Notebooks/Doohan/')

import representations as mr
import plotting as mp

import random
import math

#To make sampling at nodes more intuitive we use the following packages: 
import geopandas
from shapely.geometry import Polygon

## TOWER GEOMETRY OBJECT ##
#First we generate the octagonal shape for the towers in a geometry object:
#This is done by lists of x and y coordinates corresponding to 8th roots of unity.

# Initialize empty lists for x and y coordinates
x_coordinates = []
y_coordinates = []

# Calculate the coordinates of the 8th roots of unity and scale them:
# Calculate the angle between consecutive roots
angle_increment = 2 * math.pi / 8
scale_factor = 0.11931 #hard-coded for the corner-to-corner diameter of towers.

for i in range(8):
  angle = i * angle_increment + 2*math.pi/16 #adding constant factor to align faces.
  x = scale_factor/2 * math.cos(angle) #scale by radius of circle
  y = scale_factor/2 * math.sin(angle)
  x_coordinates.append(x)
  y_coordinates.append(y)

polygon_geom = Polygon(zip(x_coordinates, y_coordinates))
octagon = geopandas.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom])

# octagon.plot() #Check the octagon is as expected.
# octagon.sample_points(10000).plot() #This is how we sample points.


## SAMPLE RANDOM POINTS ##

def sample_maze_points(maze,n):
  ## INPUT: maze networkx object and number of points n.
  ## OUTPUT: pandas dataframe with x and y coordinates.
  #initialise arrays for x and y coordinates:
  x_list = []
  y_list = []

  for i in range(n): #iterate over the number of points to sample

    #Choose node or edge:
    ##OBS: we can adjust this here to sample nodes more frequently than edges.
    node_or_edge = random.randint(0,1) #0 if node, 1 if edge. 

    if node_or_edge == 0: #if 0, sample (x,y) from random node
      index = random.randint(0,maze.number_of_nodes()-1) #pick a random node number, subtracting 1 because python indexing starts at 0...
      #the attribute "position"  = euclidean coordinate for centre of node or edge
      centre_XY = dict(maze.nodes)[list(maze.nodes)[index]]['position'] #indexes out the position of a given node
      random_x = centre_XY[0] + octagon.sample_points(1).get_coordinates()['x'][0]
      random_y = centre_XY[1] + octagon.sample_points(1).get_coordinates()['y'][0]

    elif node_or_edge == 1: #if 1, sample (x,y) from random edge
      index = random.randint(0,maze.number_of_edges()-1) #pick a random edge number, subtracting 1 because python indexing starts at 0...
      #the attribute "position" = euclidean coordinate for centre of node or edge
      centre_XY = dict(maze.edges)[list(maze.edges)[index]]['position'] #indexes out the position of a given node
      #OBS: we need to control for the orientation being horizontal or vertical.
      #We do so by checking that the letters are the same (e.g. A1-A2), implying it's vertical.
      vertical_edge = maze.edges[list(maze.edges)[index]]['label'][0] == maze.edges[list(maze.edges)[index]]['label'][3] #The indexing is a bit ugly, but it works.

      if vertical_edge == True:
          random_x = centre_XY[0] + random.uniform(-0.022,0.022) #magic numbers here are derived from dimensions of bridges.
          random_y = centre_XY[1] + random.uniform(-0.040,0.040)
      elif vertical_edge == False:
          random_x = centre_XY[0] + random.uniform(-0.040,0.040) #magic numbers here are derived from dimensions of bridges.
          random_y = centre_XY[1] + random.uniform(-0.022,0.022)

    x_list.append(random_x) #grow our list!
    y_list.append(random_y) #grow our list!

  data = mp.pd.DataFrame(zip(x_list,y_list), columns=['x','y'])

  return data