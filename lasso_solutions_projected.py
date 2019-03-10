import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import my_toolbox.sparsity as sparse 
import my_toolbox.forwardbackward as fb


np.random.seed(seed=74) # Seed for np.random (78)
dpi = 230 # Resolution used for plotting (230 for small screen, 100 for large one)
plt.ion()

# We start by defining the characteristics of the problem: dimensions, sparsity level, etc.
data_size = 100 
data_number = round(data_size/2)
sparsity_level = 10

# We define the main components of our problem
Phi = np.random.randn(data_number,data_size)
x0 = np.sign(sparse.randn(data_size,1,sparsity_level))
y = Phi@x0
P_rand = sparse.rand_plane(data_size) # A random plane onto which we'll project the data

# We do functions which computes and display the regularization path 
def compute_reg_path(Phi, y, reg_param_grid):
    reg_path = np.empty((Phi.shape[1],0), int)
    for reg_param in reg_param_grid:
        # We choose the number of iterations to do depending on the reg_param. This is a completely custom choice, it seems to work quite well on random problems.
        if reg_param < 0.1:
            iter_nb=40000
        elif reg_param < 1:
            iter_nb=4000
        elif reg_param < 10:
            iter_nb = 1000
        else:
            iter_nb = 200
        x_reg = fb.lasso(Phi, y, reg_param, iter_nb)
        reg_path = np.concatenate((reg_path,x_reg), axis=1)
    return reg_path

def scatter_reg_path(path,limit=None,title=None):
    cm = plt.cm.get_cmap('RdYlBu')
    if limit is not None:
        plt.scatter(limit[0],limit[1],c='r',marker='x',s=150)
    if title is not None:
        _ = plt.title(title)
    plt.plot(path[0,:],path[1,:], c='black',linewidth=0.5)
    fig = plt.scatter(path[0,:], path[1,:], c=range(path.shape[1]), vmin=0, vmax=path.shape[1],  s=35, cmap=cm)
    plt.colorbar(fig)

# We put a grid on the reg_param, by taking them on a log scale and decreasing order
regp_min=-2
regp_max=2
regp_number=20
reg_param_grid = np.round(np.logspace(regp_min,regp_max,regp_number),3)[::-1]

# Let's compute the regularization path for various values of noise
noisy_vector = np.random.randn(data_number,1)
reg_path = compute_reg_path(Phi, y, reg_param_grid)
reg_path_0 = compute_reg_path(Phi, y + 1e-0 * noisy_vector, reg_param_grid)
reg_path_1 = compute_reg_path(Phi, y + 1e-1 * noisy_vector, reg_param_grid)
reg_path_2 = compute_reg_path(Phi, y + 1e-2 * noisy_vector, reg_param_grid)

# We represent the data on a plane. Let's use SVD on the noisier reg path.
# The results look less weird and ressembles the reg path in the quadratic case.
centered_points = reg_path_0 - np.mean(reg_path_0,axis=1).reshape(reg_path_0.shape[0],1)
V = np.cov(centered_points)
_, vectors = la.eig(V)
P_svd = np.real(vectors[:,0:2]).T

plt.figure(dpi=dpi)
plt.subplot(2, 2, 1)
scatter_reg_path(P_svd@reg_path,limit=P_svd@x0,title="$\sigma=0$")
plt.subplot(2, 2, 2)
scatter_reg_path(P_svd@reg_path_2,limit=P_svd@x0,title="$\sigma=0.01$")
plt.subplot(2, 2, 3)
scatter_reg_path(P_svd@reg_path_1,limit=P_svd@x0,title="$\sigma=0.1$")
plt.subplot(2, 2, 4)
scatter_reg_path(P_svd@reg_path_0,limit=P_svd@x0,title="$\sigma=1$")
plt.show()

# Parameter selection
def reg_param_selection(path,ground_truth):
    # Given a regularization path and a ground truth, returns the best regularized solution (in L2 sense)
    path_length = path.shape[1]
    reg_sol = path[:,0]
    reg_param_index = 0
    for k in np.arange(path_length):
        if la.norm(ground_truth.T - path[:,k],1) < la.norm(ground_truth.T - reg_sol,1):
            reg_sol = path[:,k]
            reg_param_index = k
    return reg_sol, reg_param_index

reg_sol, reg_param_index = reg_param_selection(reg_path,x0)
reg_sol_2, reg_param_index_2 = reg_param_selection(reg_path_2,x0)
reg_sol_1, reg_param_index_1 = reg_param_selection(reg_path_1,x0)
reg_sol_0, reg_param_index_0 = reg_param_selection(reg_path_0,x0)

_ = plt.figure(dpi=dpi)
plt.subplot(2, 2, 1)
sparse.stem(x0,"C0","ground truth")
sparse.stem(reg_sol,"C1","reg solution")
plt.title("$\sigma=0$")
plt.subplot(2, 2, 2)
sparse.stem(x0,"C0","ground truth")
sparse.stem(reg_sol_2,"C1","reg solution")
plt.title("$\sigma=0.01$")
plt.subplot(2, 2, 3)
sparse.stem(x0,"C0","ground truth")
sparse.stem(reg_sol_1,"C1","reg solution")
plt.title("$\sigma=0.1$")
plt.subplot(2, 2, 4)
sparse.stem(x0,"C0","ground truth")
sparse.stem(reg_sol_0,"C1","reg solution")
plt.title("$\sigma=1$")
plt.show()



















####################################################
############ OLD ###################################
####################################################

"""
# P_rand = sparse.rand_plane(data_size)
plt.figure(dpi=dpi)
plt.subplot(2, 2, 1)
scatter_reg_path(P_rand@reg_path,limit=P_rand@x0,title="$\sigma=0$")
plt.subplot(2, 2, 2)
scatter_reg_path(P_rand@reg_path_2,limit=P_rand@x0,title="$\sigma=0.01$")
plt.subplot(2, 2, 3)
scatter_reg_path(P_rand@reg_path_1,limit=P_rand@x0,title="$\sigma=0.1$")
plt.subplot(2, 2, 4)
scatter_reg_path(P_rand@reg_path_0,limit=P_rand@x0,title="$\sigma=1$")
plt.show()
"""
