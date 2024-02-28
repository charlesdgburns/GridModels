## A function to bring maze representations into rat in a box ##

#Input: a networkx maze representation (get_simple_maze output)
#Output: a Rat In A Box environment with walls outlining the maze.

##TODO: a Rat In A Box environment with boundaries and holes defining the maze.

## Natural language description: we want to generate walls from the maze.
# RatInABox adds walls by lines described by two coordinate points (start and end)
# The maze is made up of octagonal towers connected by bridges, where we want to exclude walls.
# Logic of the code is to assume all pairs of coordinates are walls (held in a list of tuples with start + end coordinates)
# however, if an edge coincides with a bridge, then we exclude that pair of coordinates.
# The bridge boundaries are either horizontal or vertical, which can be detected by label comparison.

#Setup

#Importing files from Peter Doohan's maze codebase:

import functions.representations as mr
import numpy as np

import ratinabox


## Geometry prep ##

# Each maze consists of octagonal towers (nodes) and bridges (edges).
# We include real-world dimensions below:

maze_width = 1.38 #total width of maze enclosure in metres.
bridge_width = 0.022 #half the width of the bridges in m; Hardcoded
bridge_length = 0.040 #half the length of the bridges in m; Hardcoded

# Walls of an octagon #

# in order to define boundaries we need vertices of our octagon.
# initialise an array with coordinates for each point in our octagon centred at (0,0) [later shifted for each tower]
points_array =  [np.zeros(2),np.zeros(2),np.zeros(2),np.zeros(2),np.zeros(2),np.zeros(2),np.zeros(2),np.zeros(2)]

# We calculate the coordinates of the 8th roots of unity and scale them:
# Calculate the angle between consecutive roots
angle_increment = 2 * np.pi / 8
scale_factor = 0.11931 #hard-coded for the corner-to-corner diameter of towers.
for i in range(8):
  angle = i * angle_increment + 2*np.pi/16 #adding constant factor to align faces with north/south/east/west.
  x = scale_factor/2 * np.cos(angle) #scale by radius of circle
  y = scale_factor/2 * np.sin(angle)
  points_array[i] = [x, y]

# Lastly, we get a generic list of walls (as formatted in RatInABox) for our octagon, centred at (0,0)
octagon_wall_list = []
for i in range(7): #all pairs except the last one
  wall_coords = [points_array[i], points_array[i+1]]
  octagon_wall_list.append(wall_coords)

octagon_wall_list.append([points_array[7], points_array[0]]) #last wall


## DEFINE FUNCTION ## 

def get_maze_env(maze):
  env = ratinabox.Environment(params = {"scale":maze_width}) #we scale the environment to fit the maze.
  #In this environment we add walls by straight lines between two points
  #e.g. env.add_wall([[x1,y1],[x2,y2]])

  ## Add walls for nodes/towers ## 

  for node in maze.nodes:
  
    #Assume we build all walls, thus we shift wall_coord_list by the node position:
    coord_shift = maze.nodes[node]['position']
    node_wall_list = np.add(octagon_wall_list, [coord_shift,coord_shift]) #a list of walls for our given node (octagonal tower)

    #Now we want to exclude the walls associated with edges.
    exclusion_list = [] #initialise
    difference_to_edge_dict = {(0,1):1, (-1,0):3, (0,-1): 5, (1,0):7} #hard-coded dictionary to index which edge to exclude.

    for e in maze.edges(node): #for each edge associated with the node (tower)
      difference = np.subtract(e[1],e[0]) #difference between coordinates of towers connected by the edge.
      exclusion_list.append(difference_to_edge_dict[(difference[0],difference[1])]) #go from dictionary to the index list 

    node_wall_list = np.delete(node_wall_list, exclusion_list ,0) #Here we exclude the walls where we have bridges!
    #BUILD THE WALLS
    for wall in node_wall_list:
      env.add_wall(wall)

  ## Add walls for edges/bridges ##

  for edge in maze.edges:

    edge_wall_list = [] #we need a list as there will be two walls per edge.
  
    #OBS: the wall coordinates will depend on whether the edge is vertical or horizontal:
    #We do so by checking that the letters of the label are the same (e.g. A1-A2), implying it's vertical. Otherwise, it's horizontal.
    vertical_edge = maze.edges[edge]['label'][0] == maze.edges[edge]['label'][3] #The indexing is a bit ugly, but it works. 
    if vertical_edge == True:
      edge_wall_list.append([[-bridge_width,-bridge_length],[-bridge_width,bridge_length]]) #left-hand-side (west) wall.
      edge_wall_list.append([[bridge_width,-bridge_length],[bridge_width,bridge_length]]) #right-hand-side (east) wall.
    elif vertical_edge == False:
      edge_wall_list.append([[-bridge_length,-bridge_width],[bridge_length,-bridge_width]]) #lower (south) wall.
      edge_wall_list.append([[-bridge_length,bridge_width],[bridge_length,bridge_width]]) #upper (north) wall.

    #Again we need to shift by the centre of the edge:
    coord_shift = maze.edges[edge]['position'] #shift standard coordinates by centre of edge.
    edge_wall_list = np.add(edge_wall_list, [coord_shift,coord_shift]) #compute coord shift

    #BUILD THE WALLS

    for wall in edge_wall_list:
      env.add_wall(wall)

  return env