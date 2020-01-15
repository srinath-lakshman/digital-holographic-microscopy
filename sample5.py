import numpy as np
import os
from skimage import io, restoration, exposure
from PIL import Image
from matplotlib import pyplot as plt

###############################################################################

y = np.arange(1,2,0.01)

x = (2*y*y)/((1/12) - (0.12*y))

plt.plot(y,x)
plt.show()
