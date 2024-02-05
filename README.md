# maze_grid_cells

The following repository is a private place to develop a series of representations of grid cells in mazes.
Final code will be hosted in the Cognitive Circuits github repository.

Here we explore the representation of space in grid-like codes, with the aim of investigating representations in a maze which is designed to decorrelate euclidean and maze path distances. 

1) Successor representation of grid cells
Inspired by Stachenfeld et al., (2014, 2017), we derive a series of representations from state transitions.
Here, grid-like codes are derived from eigendecompositions of successor representations, a state transition matrix discounted for future value.
To simplify our model, we assume a random walk policy.

2) Feedforward models (in progress)
Following Dordek et al., (2016), we consider a feedforward model from place cells input to grid cell output.
This feedforward network resembles a simple neuronal network for extracting principal components of input, therefore related to Stachenfeld et al. (2014).
In open fields, Hexagonal grids arise when activity is constrained to be non-negative.

3) more to come.
