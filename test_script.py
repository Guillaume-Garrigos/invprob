import numpy as np
import matplotlib.pyplot as plt

x = np.zeros((2,1))
x[0]=1
print(x)

def stemperso(signal):
    signal[signal==0 ] = np.nan
    plt.stem(signal)
    return

stemperso(x)
print(x)