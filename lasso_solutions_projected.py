
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import my_toolbox.sparsity as sparse 
import my_toolbox.forwardbackward as fb


np.random.seed(seed=78) # Seed for np.random
dpi = 100 # Resolution used for plotting (230 for small screen, 100 for large one)
plt.ion()

# We start by defining the characteristics of the problem: dimensions, sparsity level, etc.
data_size = 100 
data_number = round(data_size/2)
sparsity_level = 10

# We define the main components of our problem
Phi = np.random.randn(data_number,data_size)
x0 = np.sign(sparse.randn(data_size,1,sparsity_level))
P_rand = sparse.rand_plane(data_size) # A random plane onto which we'll project the data

# We do a function which computes the regularization path 
def compute_reg_path(Phi, x0, noise_level, regp_min, regp_max, regp_number):
    y = Phi@x0 + noise_level * np.random.randn(Phi.shape[0],1)
    reg_path = np.empty((Phi.shape[1],0), int)
    for reg_param in np.round(np.logspace(regp_min,regp_max,regp_number),3):
        if reg_param < 0.1:
            iter_nb=30000
        elif reg_param < 1:
            iter_nb=4000
        elif reg_param < 10:
            iter_nb = 1000
        else:
            iter_nb = 200
        x_reg = fb.lasso(Phi, y, reg_param, iter_nb)
        reg_path = np.concatenate((x_reg, reg_path), axis=1)
    return reg_path


# Let's compute the regularization path for no noise
noise_level=0
reg_path = compute_reg_path(Phi, x0, noise_level, regp_min=-2, regp_max=2, regp_number=100)
reg_path_proj = P_rand @ reg_path
x0_proj = P_rand@x0

plt.figure(dpi=dpi)
#plt.plot(reg_path_proj[0,:],reg_path_proj[1,:])
plt.scatter(reg_path_proj[0,:], reg_path_proj[1,:], c=range(reg_path_proj.shape[1]), vmin=0, vmax=20,  s=35, cmap=plt.cm.get_cmap('RdYlBu'))
sc = plt.scatter(x0_proj[0],x0_proj[1],c='r',marker='x',s=500)
plt.colorbar(sc)
plt.show()

plt.figure(dpi=dpi)
cm = plt.cm.get_cmap('RdYlBu')
x = reg_path_proj[0,:]
y = reg_path_proj[1,:]
z = range(reg_path_proj.shape[1])
sc = plt.scatter(x, y, c=z, vmin=0, vmax=100, s=35, cmap=cm)
plt.colorbar(sc)
plt.show()

# We display the regularization path (projected on a 2D space)
_, x0_proj = sparse.proj_plane(x0,u,v)
arr_proj = np.empty((2,0), int)
for i in np.arange(reg_path.shape[1]):
    _, x_proj = sparse.proj_plane(reg_path[:,i],u,v)
    arr_proj = np.concatenate((x_proj, arr_proj), axis=1)
    print(arr_proj.shape)

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
