import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

import invprob.sparse as sparse
import invprob.optim as optim
from invprob.optim import fb_lasso



np.random.seed(seed=78)  # Seed for np.random (78)
dpi = 100  # Resolution for plotting (230 for small screen, 100 for large one)
plt.ion()
folder = "scripts/../output/L1_reg/"

# We start by defining the characteristics of the problem
data_size = 100
data_number = round(data_size / 2)
sparsity_level = 10

# We define the main components of our problem
Phi = np.random.randn(data_number, data_size)
x0 = np.sign(sparse.randn(data_size, 1, sparsity_level))
noisy_vector = np.random.randn(data_number, 1)
y = Phi@x0 + 0 * noisy_vector

# We choose a decreasing sequence of reg_param
iter_nb = 40000
regp_min = -2
regp_max = 2
regp_number = iter_nb
reg_param_grid = np.round(np.logspace(regp_min, regp_max, regp_number), 3)
reg_param_grid = 1 / (np.arange(regp_number)+1)

x_sol, details = fb_lasso(Phi, y, reg_param_grid, iter_nb, verbose=True)

_ = plt.figure(dpi=dpi)
sparse.stem(x0, "C0", "ground truth")
sparse.stem(details["iterate_path"][:, 3000], "C1", "reg solution")


#########################################
# This is for production only
import importlib
importlib.reload(optim)
import invprob.optim as optim
from invprob.optim import fb_lasso
#########################################