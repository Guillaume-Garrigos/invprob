
#%%
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import Image, display

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

import my_toolbox.sparsity as sparse 
import my_toolbox.forwardbackward as fb

#%%
np.random.seed(seed=78) # Seed for np.random
dpi = 100 # Resolution used for plotting (230 for small screen, 100 for large one)


#%%
# We start by defining the characteristics of the problem: dimensions, sparsity level, etc.
data_size = 100 
data_number = round(data_size/2)
sparsity_level = 10
noise_level = 1e-2*0


#%%
# We define the main components of our problem
Phi = np.random.randn(data_number,data_size)
x0 = np.sign(sparse.randn(data_size,1,sparsity_level))
noise = noise_level * np.random.randn(data_number,1)
y = Phi@x0 + noise


#%%
def compute_solution(reg_param):
    reg_param = round(reg_param,1)
    if reg_param > 10:
        iter_nb = 200
    else:
        if reg_param > 1:
            iter_nb = 1000
        else:
            iter_nb = 3000
    _ = plt.figure(dpi=dpi)
    x_reg = fb.lasso(Phi, y, reg_param, iter_nb)
    _=plt.title(r"Regularisation parameter $\lambda=$"+str(reg_param))
    sparse.stem(x0,"C0","ground truth")
    sparse.stem(x_reg,"C1","reg solution")
    plt.show()
    print(str(reg_param))


#%%
interactive(compute_solution,reg_param=(1,100,1))


#%%
def show_solution(reg_param):
    reg_param = round(reg_param + 0.0,1)
    display(Image(filename='output/L1_reg/reg_sol_'+str(reg_param)+'.png'))


#%%
interactive(show_solution,reg_param=(0.1,100,0.1))


#%%



