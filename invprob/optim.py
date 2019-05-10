import numpy as np
from numpy import linalg as la
import invprob.sparse as sparse


def fb_lasso(A, y, reg_param, iter_nb, x_ini=None, verbose=False):
    ''' Use the Forward-Backward algorithm to find a minimizer of:
             reg_param*norm(x,1) + 0.5*norm(Ax-y,2)**2
        Eventually outputs the functional values and support of the iterates
        while running the method
        reg_param is either a number, in which case we use it all along the iterations
                  or a sequence of size iter_nb
    '''
    # Manage optional input/output
    if verbose:  # Optional output
        print("new")
        regret = np.zeros(iter_nb)
        support = np.zeros(iter_nb)
        path = np.zeros((A.shape[1], iter_nb))
    if x_ini is not None:  # Optional initialization
        x = x_ini
    else:
        x = np.zeros((A.shape[1], 1))
    if isinstance(reg_param, (int, float)):  # Fixed or not parameter
        param = reg_param * np.ones(iter_nb)
    else:
        param = reg_param


    # The core of the algorithm
    stepsize = 0.8 * 2 / (la.norm(A, 2)**2)
    for k in range(iter_nb):
        x = x - stepsize * A.T@(A@x - y)
        x = sparse.soft_thresholding(x, param[k] * stepsize)
        if verbose:
            regret[k] = 0.5 * la.norm(A@x - y, 2)**2 + param[k] * la.norm(x, 1)
            support[k] = sparse.norm0(x)
            path[:, k] = x.reshape((x.shape[0]))

    # Output
    if verbose:
        details = {
            "function_value": regret,
            "iterate_support": support,
            "iterate_path": path
        }
        return x, details
    else:
        return x
