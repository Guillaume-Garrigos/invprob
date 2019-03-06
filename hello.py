# First modification
# last test
N=6
s=2

x = np.random.randn(N)
index = np.random.permutation(N)
sub_index = index[0:s]

c = [x[i] for i in sub_index]
c
c[0]=0
x
a= np.ones((10))
ind= np.random.permutation(10)
ind=ind[0:4];
a[np.ix_(ind)]*=0
a = a.reshape((5,2))
def test_function(N,M): 
    x = np.random.randn(N*M)
    x = x.reshape((N,M))
    return x

test_function(3,2)


import matplotlib.pyplot as plt
x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
plt.plot(x, np.sin(x))       # Plot the sine of each x point
plt.show()                   # Display the plot



