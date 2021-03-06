import os.path
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
import invprob.sparse as sparse
from invprob.optim import fb_lasso

#########################################
# This is for production only
import importlib
importlib.reload(sparse)
importlib.reload(fb)
#########################################

np.random.seed(seed=78)  # Seed for np.random
dpi = 100  # Resolution for plotting (230 for small screen, 100 for large one)
plt.ion()
folder = "scripts/../output/L1_reg/"

# We start by defining the characteristics of the problem
data_size = 100
data_number = round(data_size / 2)
sparsity_level = 10
noise_level = 1e-2 * 0

# We define the main components of our problem
Phi = np.random.randn(data_number, data_size)
x0 = np.sign(sparse.randn(data_size, 1, sparsity_level))
noise = noise_level * np.random.randn(data_number, 1)
y = Phi@x0 + noise

# Let's compare the ground truth with the pseudo inverse solution
x_pinv = la.lstsq(Phi, y, rcond=None)[0]
_ = plt.figure(dpi=dpi)
sparse.stem(x0, "C0", "ground truth")
sparse.stem(x_pinv, "C1", "pinv solution")
plt.show()

# Let's compare the ground truth with the solution of the LASSO
# (computed with the Forward-Backward algorithm)
reg_param = 0.01
iter_nb = 40000

x_reg = fb_lasso(Phi, y, reg_param, iter_nb)
_ = plt.figure(dpi=dpi)
sparse.stem(x0, "C0", "ground truth")
sparse.stem(x_reg, "C1", "reg solution")
plt.show()

# We look at what happens during the iterations of the algorithm
x_reg, details = fb_lasso(Phi, y, reg_param, iter_nb, verbose=True)
plt.figure(dpi=dpi)
plt.title(r"Evolution of $f(x_n)$")
plt.plot(details.get("function_value"))
plt.figure(dpi=dpi)
plt.title(r"Evolution of supp$(x_n)$")
plt.plot(details.get("iterate_support"))
plt.show()

# Now we generate the regularization path
# Quite expensive in time depending on the parameters!
def compute_reg_path(Phi, y, reg_param_grid):
    print("Computing the regularization path")
    reg_path = np.empty((Phi.shape[1], 0), int)
    x_ini = np.zeros((Phi.shape[1], 1))
    for reg_param in reg_param_grid:
        ''' We choose the number of iterations to do depending on the reg_param.
        This is a completely custom choice, it seems to work quite well
        on random problems.
        '''
        if reg_param < 0.1:
            iter_nb = 40000
        elif reg_param < 1:
            iter_nb = 4000
        elif reg_param < 10:
            iter_nb = 1000
        else:
            iter_nb = 200
        # We use a warm restart approach:
        # for each problem we use the solution of the previous problem
        # as a starting point
        x_reg = fb_lasso(Phi, y, reg_param, iter_nb, x_ini=x_ini)
        x_ini = x_reg
        reg_path = np.concatenate((reg_path, x_reg), axis=1)
    return reg_path

regp_min = -2
regp_max = 2
regp_number = 200
reg_param_grid = np.round(np.logspace(regp_min, regp_max, regp_number), 3)[::-1]

if os.path.isfile(folder + 'reg_path_noiseless.npy'):
    reg_path = np.load(folder + 'reg_path_noiseless.npy')
    if reg_path.shape[1] != regp_number:  # Previous but different experiment
        reg_path = compute_reg_path(Phi, y, reg_param_grid)
        np.save(folder + 'reg_path_noiseless.npy', reg_path)
else:
    reg_path = compute_reg_path(Phi, y, reg_param_grid)
    np.save(folder + 'reg_path_noiseless.npy', reg_path)

# We save the reg path as many image files and as an animated gif
# This is the name under which we save the data
file_name = folder + 'reg_path_noiseless'
# We concatenate conveniently x0 and reg_path in such a way that for every frame
# we plot two signals: x0 and reg_path[param]
paths = np.stack((np.repeat(x0, regp_number, axis=1), reg_path),
                 axis=2)
# We chose a title for every frame we'll plot
title_grid = [r"Ground truth $x_0$ vs regularised solution $x_\lambda$ " +
              "for $\lambda$=" + str(param) for param in reg_param_grid]

plt.ioff()
plt.figure(dpi=dpi)
options = {"animation": False,  # What we wanna save and how
           "frames": False,
           "interval": 100,
           "file_name": file_name}
sparse.save_stem_gif(paths, reg_param_grid, title_grid, options)
