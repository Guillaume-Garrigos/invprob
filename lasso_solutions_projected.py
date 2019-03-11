import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import os.path
import my_toolbox.sparsity as sparse
import my_toolbox.forwardbackward as fb

np.random.seed(seed=74)  # Seed for np.random (78)
dpi = 230  # Resolution for plotting (230 for small screen, 100 for large one)
plt.ion()
folder = "output/L1_reg/"

# We start by defining the characteristics of the problem
data_size = 100
data_number = round(data_size / 2)
sparsity_level = 10

# We define the main components of our problem
Phi = np.random.randn(data_number, data_size)
x0 = np.sign(sparse.randn(data_size, 1, sparsity_level))
y = Phi@x0
noisy_vector = np.random.randn(data_number, 1)

# We put a grid on the reg_param, by taking it on a log scale and decreasing
regp_min = -2
regp_max = 2
regp_number = 100
reg_param_grid = np.round(np.logspace(regp_min, regp_max, regp_number), 3)

# Let's compute the regularization path for various values of noise
def compute_reg_path(Phi, y, reg_param_grid):
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
        x_reg = fb.lasso(Phi, y, reg_param, iter_nb, x_ini=x_ini)
        x_ini = x_reg
        reg_path = np.concatenate((reg_path, x_reg), axis=1)
    return reg_path

if os.path.isfile(folder + 'reg_path_example.npy'):
    reg_path_examples = np.load(folder + 'reg_path_examples.npy')
    reg_path = reg_path_examples[:, :, 0]
    reg_path_2 = reg_path_examples[:, :, 1]
    reg_path_1 = reg_path_examples[:, :, 2]
    reg_path_0 = reg_path_examples[:, :, 3]
else:
    reg_path = compute_reg_path(Phi, y, reg_param_grid)
    reg_path_0 = compute_reg_path(Phi, y + 1e-0 * noisy_vector, reg_param_grid)
    reg_path_1 = compute_reg_path(Phi, y + 1e-1 * noisy_vector, reg_param_grid)
    reg_path_2 = compute_reg_path(Phi, y + 1e-2 * noisy_vector, reg_param_grid)
    reg_path_examples = np.dstack((reg_path, reg_path_2,
                                   reg_path_1, reg_path_0))
    np.save(folder + 'reg_path_examples.npy', reg_path_examples)

# We compute the best parameter given the ground truth
def reg_param_selection(path, ground_truth):
    '''Given a regularization path and a ground truth,
    returns the best regularized solution (in L2 sense)
    '''
    path_length = path.shape[1]
    path_dimension = path.shape[0]
    reg_sol = path[:, 0, None]
    reg_param_index = 0
    for k in np.arange(path_length):
        if la.norm(ground_truth - path[:, k, None]) \
                < la.norm(ground_truth - reg_sol):
            reg_sol = path[:, k, None]
            reg_param_index = k
    return reg_sol, reg_param_index

reg_sol, reg_param_index = reg_param_selection(reg_path, x0)
reg_sol_2, reg_param_index_2 = reg_param_selection(reg_path_2, x0)
reg_sol_1, reg_param_index_1 = reg_param_selection(reg_path_1, x0)
reg_sol_0, reg_param_index_0 = reg_param_selection(reg_path_0, x0)

# We represent the selected solutions
_ = plt.figure(dpi=dpi)
plt.subplot(2, 2, 1)
sparse.stem(x0, "C0", "ground truth")
sparse.stem(reg_sol, "C1", "reg solution")
plt.title("$\sigma=0$")
plt.subplot(2, 2, 2)
sparse.stem(x0, "C0", "ground truth")
sparse.stem(reg_sol_2, "C1", "reg solution")
plt.title("$\sigma=0.01$")
plt.subplot(2, 2, 3)
sparse.stem(x0, "C0", "ground truth")
sparse.stem(reg_sol_1, "C1", "reg solution")
plt.title("$\sigma=0.1$")
plt.subplot(2, 2, 4)
sparse.stem(x0, "C0", "ground truth")
sparse.stem(reg_sol_0, "C1", "reg solution")
plt.title("$\sigma=1$")

# We represent the data on a plane. Let's use SVD on the noisier reg path.
# The results ressembles the reg path in the quadratic case.
def get_pca_projector(data):
    ''' Given a set of data points (stored in columns of a 2D np.array)
    returns the projection matrix corresponding to the 2 more relevant
    directions
    '''
    centered_points = data \
        - np.mean(data, axis=1).reshape(data.shape[0], 1)
    V = np.cov(centered_points)
    _, vectors = la.eig(V)
    return np.real(vectors[:, 0:2]).T

P_svd = get_pca_projector(reg_path_0)
# Alternative: just using a random projection
# P_rand = sparse.rand_plane(data_size)

plt.figure(dpi=dpi)
plt.subplot(2, 2, 1)
scatter_reg_path(P_svd@reg_path,
                 title="$\sigma$=0, $\lambda$=" +
                 str(reg_param_grid[reg_param_index]),
                 limit=np.concatenate((P_svd@x0, P_svd@reg_sol), axis=1))
plt.subplot(2, 2, 2)
scatter_reg_path(P_svd@reg_path_2,
                 title="$\sigma$=0.01, $\lambda$=" +
                 str(reg_param_grid[reg_param_index_2]),
                 limit=np.concatenate((P_svd@x0, P_svd@reg_sol_2), axis=1))
plt.subplot(2, 2, 3)
scatter_reg_path(P_svd@reg_path_1,
                 title="$\sigma$=0.1, $\lambda$=" +
                 str(reg_param_grid[reg_param_index_1]),
                 limit=np.concatenate((P_svd@x0, P_svd@reg_sol_1), axis=1))
plt.subplot(2, 2, 4)
scatter_reg_path(P_svd@reg_path_0,
                 title="$\sigma$=1, $\lambda$=" +
                 str(reg_param_grid[reg_param_index_0]),
                 limit=np.concatenate((P_svd@x0, P_svd@reg_sol_0), axis=1))
plt.show()

# Now we try to catch what is the rate of convergence, which is
# the value of norm(x0 - x_lambda) and lambda when the noise goes to zero.
batch_nb = 20
noise_level_grid = np.round(np.logspace(-2, 0, batch_nb), 3)

if os.path.isfile(folder + 'reg_path_rates_examples.npy'):
    import_data = np.load(folder + 'reg_path_rates_examples.npy')
    noise_level_grid = import_data[:, :, 0].T
    param_value = import_data[:, :, 1].T
    dist_to_truth = import_data[:, :, 2].T
else:
    # We run a heavy computation
    param_value = np.empty([batch_nb, 1], dtype=int)
    dist_to_truth = np.zeros((batch_nb, 1))

    for k in np.arange(batch_nb):
        reg_path_temp = compute_reg_path(Phi,
                                         y + noise_level_grid[k] * noisy_vector,
                                         reg_param_grid)
        reg_sol_temp, reg_param_index_temp = reg_param_selection(reg_path_temp, x0)
        param_value[k] = reg_param_grid[reg_param_index_temp]
        dist_to_truth[k] = la.norm(reg_sol_temp - x0)

    reg_path_rates_examples = np.dstack((noise_level_grid,
                                        param_value,
                                        dist_to_truth))
    np.save(folder + 'reg_path_rates_examples.npy',
            reg_path_rates_examples)

plt.figure(dpi=dpi)
plt.title(r"Evolution of $\lambda(\sigma)$ in function of $\sigma$")
plt.loglog(noise_level_grid, param_value)
plt.figure(dpi=dpi)
plt.title(r"Evolution of $\Vert x_{\lambda_\sigma} - x_0 \Vert$ in function of $\sigma$")
plt.loglog(noise_level_grid, dist_to_truth)
plt.show()
