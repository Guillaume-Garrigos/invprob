
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
noise_level = 1e-2*0

# We define the main components of our problem
Phi = np.random.randn(data_number,data_size)
x0 = np.sign(sparse.randn(data_size,1,sparsity_level))
noise = noise_level * np.random.randn(data_number,1)
y = Phi@x0 + noise

_ = plt.figure(dpi=dpi)
sparse.stem(x0,"C0","ground truth")
plt.show()

# Let's compute all the solution of the LASSO (computed with the Forward-Backward algorithm) for various values of reg_param
arr = np.empty((data_size,0), int)
arr = np.concatenate((x0, arr), axis=1)
for reg_param in np.round(np.linspace(0.11,100,99),1):
    if reg_param < 1:
        iter_nb=3000
    elif reg_param < 10:
        iter_nb = 1000
    else:
        iter_nb = 200
    x_reg = fb.lasso(Phi, y, reg_param, iter_nb)
    arr = np.concatenate((x_reg, arr), axis=1)
#np.save("data_lasso_solutions", arr)

# We display the regularization path (projected on a 2D space)
u,v = sparse.rand_plane(data_size)
arr_proj = np.empty((2,0), int)
for i in np.arange(arr.shape[1]):
    _, x_proj = sparse.proj_plane(arr[:,i],u,v)
    arr_proj = np.concatenate((x_proj, arr_proj), axis=1)

plt.figure(dpi=dpi)
plt.plot(arr_proj[0,:],arr_proj[1,:])
plt.scatter(arr_proj[0,:],arr_proj[1,:],c=np.arange(arr_proj.shape[1]))
plt.show()

