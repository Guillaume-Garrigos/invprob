import numpy as np
from numpy import linalg as la
import invprob.sparse as sparse


def fb_lasso(A, y, reg_param, iter_nb, x_ini=None, inertia=False, verbose=False):
    ''' Use the Forward-Backward algorithm to find a minimizer of:
             reg_param*norm(x,1) + 0.5*norm(Ax-y,2)**2
        Eventually outputs the functional values and support of the iterates
        while running the method
        reg_param is either a number, in which case we use it all along the iterations
                  or a sequence of size iter_nb
    '''
    # Manage optional input/output
    if verbose:  # Optional output
        regret = np.zeros(iter_nb)
        sparsity = np.zeros(iter_nb)
        support = []
        path = np.zeros((A.shape[1], iter_nb))
    if x_ini is not None:  # Optional initialization
        x = x_ini
    else:
        x = np.zeros((A.shape[1], 1))
    if isinstance(reg_param, (int, float)):  # Fixed or not parameter
        param = reg_param * np.ones(iter_nb)
    else:
        param = reg_param
    if inertia:
        alpha = [k/(k+3) for k in np.arange(iter_nb)] # asymptotically equivalent to Nesterov
    else:
        alpha = np.zeros(iter_nb) # no inertia

    # The core of the algorithm
    stepsize = 0.5 * 2 / (la.norm(A, 2)**2)
    T = A.T@A
    ATy = A.T@y
    gradient = lambda x: x - stepsize*(T@x - ATy)
    forward_backward = lambda x, param: sparse.soft_thresholding(gradient(x), param*stepsize)
    x_old = x
    for k in range(iter_nb):
        if verbose:
            regret[k] = 0.5 * la.norm(A@x - y, 2)**2 + param[k] * la.norm(x, 1)
            support.append( tuple(np.where(np.abs(x) > 1e-15)[0]) )
            sparsity[k] = len(support[k])
            path[:, k] = x.reshape((x.shape[0]))
        x, x_old = forward_backward( (1+alpha[k])*x - alpha[k]*x_old, param[k] ), x
        
    # Output
    if verbose:
        details = {
            "function_value": regret,
            "iterate_support": support,
            "iterate_sparsity": sparsity,
            "iterate_path": path
        }
        return x, details
    else:
        return x

def cp_lasso(A, y, iter_nb, stepsize_factor=None, x_ini=None, inertia=False, verbose=False):
    ''' Use the Chambolle-Pock algorithm to find a minimizer of:
             norm(x,1) over the constraint Ax=y
        Eventually outputs the functional values and support of the iterates
        while running the method
    '''
    # Manage optional input/output
    if verbose:  # Optional output
        objective_gap = np.zeros(iter_nb)
        constraint_gap = np.zeros(iter_nb)
        sparsity = np.zeros(iter_nb)
        support = []
        path = np.zeros((A.shape[1], iter_nb))
    if x_ini is not None:  # Optional initialization
        x = x_ini
    else:
        x = np.zeros((A.shape[1], 1))
    if stepsize_factor is not None:  # Optional
        stepsize_factor = stepsize_factor
    else:
        stepsize_factor = 1
    if inertia:
        alpha = [k/(k+3) for k in np.arange(iter_nb)] # asymptotically equivalent to Nesterov
    else:
        alpha = np.zeros(iter_nb) # no inertia

    # The core of the algorithm
    stepsize = stepsize_factor / la.norm(A, 2) # sigma and tau in the C-P paper or Condat
    u = A@x
    x_extrap = x
    for k in range(iter_nb):
        if verbose:
            objective_gap[k] = la.norm(x, 1)
            constraint_gap[k] = la.norm(A@x - y, 2)
            support.append( tuple(np.where(np.abs(x) > 1e-15)[0]) )
            sparsity[k] = len(support[k])
            path[:, k] = x.reshape((x.shape[0]))
        
        u = u + stepsize*((A@x_extrap) - y)
        x_old = x
        x = sparse.soft_thresholding( x - stepsize*A.T@u , stepsize)
        x_extrap = 2*x - x_old
        
    # Output
    if verbose:
        details = {
            "objective_gap": objective_gap,
            "constraint_gap": constraint_gap,
            "iterate_support": support,
            "iterate_sparsity": sparsity,
            "iterate_path": path
        }
        return x, details
    else:
        return x
