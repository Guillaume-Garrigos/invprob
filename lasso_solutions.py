
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

# Let's compare the ground truth with the pseudo inverse solution
x_pinv = la.lstsq(Phi,y,rcond=None)[0]
_ = plt.figure(dpi=dpi)
sparse.stem(x0,"C0","ground truth")
sparse.stem(x_pinv,"C1","pinv solution")
plt.show()

# Let's compare the ground truth with the solution of the LASSO (computed with the Forward-Backward algorithm)
reg_param = 0.1
iter_nb = 3000

x_reg = fb.lasso(Phi, y, reg_param, iter_nb)
_ = plt.figure(dpi=dpi)
sparse.stem(x0,"C0","ground truth")
sparse.stem(x_reg,"C1","reg solution")
plt.show()

# We look at what happens during the iterations of the algorithm
x_reg, details = fb.lasso(Phi, y, reg_param, iter_nb, verbose=True)
plt.figure(dpi=dpi); plt.title(r"Evolution of $f(x_n)$")
plt.plot(details.get("function_value")) 
plt.figure(dpi=dpi); plt.title(r"Evolution of supp$(x_n)$")
plt.plot(details.get("iterate_support"))
plt.show()

# Now we generate and save images corresponding to the regularized solutions for various values for reg_param
# Quite expensive in time depending on the parameters!
_ = plt.figure(dpi=dpi)
for reg_param in np.round(np.linspace(10.1,100,900),1):
    x_reg = fb.lasso(Phi, y, reg_param, iter_nb=200)
    _=plt.title(r"Regularisation parameter $\lambda=$"+str(reg_param))
    sparse.stem(x0,"C0","ground truth")
    sparse.stem(x_reg,"C1","reg solution")
    plt.savefig('output/L1_reg/reg_sol_'+str(reg_param)+'.png',bbox_inches='tight')
    plt.clf()
print("done")
for reg_param in np.round(np.linspace(1.1,10,90),1):
    x_reg = fb.lasso(Phi, y, reg_param, iter_nb=1000)
    _=plt.title(r"Regularisation parameter $\lambda=$"+str(reg_param))
    sparse.stem(x0,"C0","ground truth")
    sparse.stem(x_reg,"C1","reg solution")
    plt.savefig('output/L1_reg/reg_sol_'+str(reg_param)+'.png',bbox_inches='tight')
    plt.clf()
print("done")
for reg_param in np.round(np.linspace(0.11,1,9),1):
    x_reg = fb.lasso(Phi, y, reg_param, iter_nb=3000)
    _=plt.title(r"Regularisation parameter $\lambda=$"+str(reg_param))
    sparse.stem(x0,"C0","ground truth")
    sparse.stem(x_reg,"C1","reg solution")
    plt.savefig('output/L1_reg/reg_sol_'+str(reg_param)+'.png',bbox_inches='tight')
    plt.clf()
print("done")

# Super super expensive
for reg_param in np.round(np.linspace(0.11,100,999),1):
    if reg_param < 1:
        iter_nb=3000
    elif reg_param < 10:
        iter_nb = 1000
    else:
        iter_nb = 200
    x_reg = fb.lasso(Phi, y, reg_param, iter_nb)
    _=plt.title(r"Regularisation parameter $\lambda=$"+str(reg_param))
    sparse.stem(x0,"C0","ground truth")
    sparse.stem(x_reg,"C1","reg solution")
    plt.savefig('output/L1_reg/reg_sol_'+str(reg_param)+'.png',bbox_inches='tight')
    plt.clf()
print("done")




