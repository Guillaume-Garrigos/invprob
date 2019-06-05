import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

import invprob.sparse as sparse
import invprob.optim as optim
from invprob.optim import fb_lasso



np.random.seed(seed=74)  # Seed for np.random (74)
dpi = 230  # Resolution for plotting (230 for small screen, 100 for large one)
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

# We solve the noiseless problem
iter_nb = 10000
exp_decay = 0.1  # The smaller the exponent, the faster is the algorithm
reg_param_grid = 1 / (np.arange(iter_nb)+1)**exp_decay

x_sol = fb_lasso(Phi, y, reg_param_grid, iter_nb, verbose=False)
_ = plt.figure(dpi=dpi)
sparse.stem(x0, "C0", "ground truth")
sparse.stem(x_sol, "C1", "inverse solution")

# We compute the best parameter given the ground truth
def reg_param_selection(path, ground_truth):
    '''Given a regularization path and a ground truth,
    returns the best regularized solution (in L2 sense)
    '''
    path_length = path.shape[1]
    reg_sol = path[:, 0, None]
    reg_param_index = 0
    for k in np.arange(path_length):
        if la.norm(ground_truth - path[:, k, None]) \
                < la.norm(ground_truth - reg_sol):
            reg_sol = path[:, k, None]
            reg_param_index = k
    return reg_sol, reg_param_index


# We solve the noisy problem
y = Phi@x0 + 0.01 * noisy_vector
iter_nb = 1000
exp_decay = 0.1  # The smaller the exponent, the faster is the algorithm
reg_param_grid = 1 / (np.arange(iter_nb)+1)**exp_decay

x_sol2 = fb_lasso(Phi, y, 0.001, iter_nb=100000)
x_reg2 = fb_lasso(Phi, y, 0.87, iter_nb=200)

x_sol, details = fb_lasso(Phi, y, reg_param_grid, iter_nb, verbose=True)
reg_path = details["iterate_path"]
x_reg, reg_param_index = reg_param_selection(reg_path, x0)

_ = plt.figure(dpi=dpi)
sparse.stem(x0, "C0", "ground truth")
sparse.stem(x_sol, "C2", "inv diag solution")
sparse.stem(x_sol2, "C1", "inv tikh solution")

_ = plt.figure(dpi=dpi)
sparse.stem(x0, "C0", "ground truth")
sparse.stem(x_reg, "C2", "reg diag solution")
sparse.stem(x_reg2, "C1", "reg tikh solution")






#########################################
# This is for production only
import importlib
importlib.reload(optim)
import invprob.optim as optim
from invprob.optim import fb_lasso
#########################################
