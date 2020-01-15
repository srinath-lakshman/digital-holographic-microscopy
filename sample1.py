import numpy as np
import os
from skimage import io, restoration, exposure
from PIL import Image
from matplotlib import pyplot as plt

###############################################################################

#l = np.arange(0*np.pi,10*np.pi,(8/7)*np.pi)
l = [0*np.pi, (1/4)*np.pi, (1/2)*np.pi, (7/4)*np.pi, (2)*np.pi, (2+(1/4))*np.pi, (2+(2/4))*np.pi]
# l = [0*np.pi, (1/4)*np.pi, (1/2)*np.pi, (2)*np.pi]
n = range(0,len(l))
m = np.unwrap(l)

plt.scatter(n,l)
plt.scatter(n,m)

plt.plot(n,l)
plt.plot(n,m)

plt.show()

axis = -1

p = np.asarray(l)
nd = p.ndim
dd = np.diff(p, axis=axis)
slice1 = [slice(None, None)]*nd
slice1[axis] = slice(1,None)
slice1 = tuple(slice1)

ddmod = np.mod(dd+np.pi, 2*np.pi) - np.pi
# _nx.copyto(ddmod, pi, where=(ddmod == -pi) & (dd > 0))
# ph_correct = ddmod - dd

print(l)
print(m)

#plt.show()
