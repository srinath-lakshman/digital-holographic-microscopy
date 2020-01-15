import numpy as np
import os
from skimage import io, restoration, exposure
from PIL import Image
from matplotlib import pyplot as plt

from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt

###############################################################################

window = signal.gaussian(51, std=7)
plt.plot(window)
plt.title(r"Gaussian window ($\sigma$=7)")
plt.ylabel("Amplitude")
plt.xlabel("Sample")

plt.show()

print(np.shape(window))

# x = np.zeros(50)
#
# for i in np.arange(0,20):
#     x[i] = 0
#
# for i in np.arange(20,50):
#     print(i)
#     x[i] = 30 - np.sqrt((30*30)-((i*i) - (20*20)))
#
# y = np.mod(x,2*np.pi)

# def gaussian(x, mu, sig):
#     return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
#
# # x = np.arange(-50,51)
# #
# # plot(x_values)
# count = -1
#
# x = np.linspace(-3, 3, 120)
# y = np.zeros(len(x))
#
# for mu, sig in [(0, 2), (-1, 1), (2, 3)]:
#     count = count + 1
#     y[count] = gaussian(x[count], mu, sig)
#
# plt.plot(x,y)
# #plt.plot(y)
# plt.show()
