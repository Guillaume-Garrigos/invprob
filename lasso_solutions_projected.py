
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

# Let's try to visualize the solution of the LASSO (computed with the Forward-Backward algorithm)
reg_param = 0.1
iter_nb = 3000

u,v = sparse.rand_plane(data_size)
arr = np.empty((2,0), int)
for reg_param in np.round(np.linspace(11,100,20),1):
    if reg_param < 1:
        iter_nb=3000
    elif reg_param < 10:
        iter_nb = 1000
    else:
        iter_nb = 200
    x_reg = fb.lasso(Phi, y, reg_param, iter_nb)
    _, x_p = sparse.proj_plane(x_reg,u,v)
    arr = np.concatenate((x_p, arr), axis=1)
plt.figure(dpi=dpi)
plt.plot(arr[0,:],arr[1,:])
plt.scatter(arr[0,:],arr[1,:],c=np.arange(arr.shape[1]))
plt.show()

x = np.random.random(10)
y = np.random.random(10)
c = np.arange(10)

plt.scatter(x, y, c=c, s=500)
plt.show()
