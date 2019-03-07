
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import my_toolbox.sparsity as sparse 
import my_toolbox.forwardbackward as fb

###### This is for production only ######
import importlib
importlib.reload(sparse) 
importlib.reload(fb) 
#########################################
np.random.seed(seed=78) # Seed for np.random
dpi = 230 # Resolution used for plotting (230 for small screen, 100 for large one)

# We start by defining the characteristics of the problem: dimensions, sparsity level, etc.
data_size = 100 
data_number = round(data_size/2)
sparsity_level = 10
noise_level = 0.001*0

# We define the main components of our problem
Phi = np.random.randn(data_number,data_size)
x0 = np.sign(sparse.randn(data_size,1,sparsity_level))
noise = noise_level * np.random.randn(data_number,1)
y = Phi@x0 + noise

# Let's compute all the solution of the LASSO (computed with the Forward-Backward algorithm) for various values of reg_param
sol_list = np.empty((data_size,0), int)
for reg_param in np.round(np.linspace(0.01,1,99),2):
    if reg_param < 0.1:
        iter_nb=40000
    elif reg_param < 1:
        iter_nb=4000
    elif reg_param < 10:
        iter_nb = 1000
    else:
        iter_nb = 200
    x_reg = fb.lasso(Phi, y, reg_param, iter_nb)
    sol_list = np.concatenate((x_reg, sol_list), axis=1)
#np.save("data_lasso_solutions", arr)
#arr = np.load("data_lasso_solutions.npy")

# We display the regularization path (projected on a 2D space)
u,v = sparse.rand_plane(data_size)
_, x0_proj = sparse.proj_plane(x0,u,v)
arr_proj = np.empty((2,0), int)
for i in np.arange(sol_list.shape[1]):
    _, x_proj = sparse.proj_plane(sol_list[:,i],u,v)
    arr_proj = np.concatenate((x_proj, arr_proj), axis=1)

plt.figure(dpi=dpi)
plt.plot(arr_proj[0,:],arr_proj[1,:])
plt.scatter(arr_proj[0,:],arr_proj[1,:],c=np.arange(arr_proj.shape[1]))
plt.scatter(x0_proj[0],x0_proj[1],c='r',marker='x',s=500)
plt.show()


M = np.mean(sol_list,axis=1)
C = sol_list.T - M
V = np.cov(C.T)
_, vectors = la.eig(V)
P = vectors.T.dot(sol_list)
sol_proj_list = np.real(P[0:2,:])
P0 = vectors.T.dot(x0)
x0_proj = np.real(P0[0:2,:])

plt.figure(dpi=dpi)
plt.plot(sol_proj_list[0,:],sol_proj_list[1,:])
plt.scatter(sol_proj_list[0,:],sol_proj_list[1,:],c=np.arange(sol_proj_list.shape[1]))
plt.scatter(x0_proj[0],x0_proj[1],c='r',marker='x',s=500)
plt.show()
