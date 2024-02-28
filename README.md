# maze_grid_cells

WORK IN PROGRESS

The following repository is a private place to develop a series of representations of grid cells in mazes.
Final code will eventually be hosted in the Cognitive Circuits github repository.

We inherit representations of Mazes from Peter Doohan's previous work, importing these to the RatInABox environment for simulating a mouse in a maze.

Here we explore the representation of space in grid-like codes, with the aim of investigating representations in a maze which is designed to de-correlate euclidean and maze path distances. 

We include jupyter notebooks for each representation as a demo before writing a final main script which allows us to generate a given representation for a given grid cell

1) Successor representation of grid cells
Inspired by Stachenfeld et al., (2014, 2017), we derive a series of representations from state transitions.
Here, grid-like codes are derived from eigendecompositions of successor representations, a state transition matrix discounted for future value.
To simplify our model, we assume a random walk policy.

2) Feedforward models (in progress)
Following Dordek et al., (2016), we consider a feed-forward model from place cells input to grid cell output.
This feed-forward network resembles a simple neuronal network for extracting principal components of input, therefore related to Stachenfeld et al. (2014).
In open fields, Hexagonal grids arise when weights are constrained to be non-negative.

3) more to come.

--- Dependencies ---

matplotlib, networkX, Pandas, Numpy
