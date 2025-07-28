# Script folder description

The following folder is intended to store RNNs with different architectures and training procedures.

All RNN architectures (and with their respective objectives) will be trained from pre-stored trajectories. 
This will either be from behavioural data, or from RatInABox generated trajectories.

Since path integration models of hippocampal formation can differ a lot we take a slightly different coding approach and have dedicated scripts which contain everything from the model, the loss functions, and the training loops.

## vel2pc_RNN.py
This script is meant to repliate Sorscher et al.'s (2023) grid cell models. These take only velocity as input and 
- for example, the Sorscher et al. (2023) style models take only velocity as input and their output is a predicted activation strength of hippocampal place cells.
- Note further, that this could be re-cast as prediction of future input if the inputs include place-cell inputs from HPC.

## continuous_TEM.py
This script is meant to introduce TEM-like modelling, where place-cell like inputs are themselves learnt using sensory inputs (instead of assumed to exist).


## 2D_RNN.py
This script is meant to introduce a biological constraint that is aimed at capturing global modularisation.
Here we aim to show that constraints on weights described by Khona et al. (2025) can be derived from biologically plausible constraints.

