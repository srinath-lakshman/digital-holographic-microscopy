import numpy as np
import os
from skimage import io, restoration, exposure
from PIL import Image
from matplotlib import pyplot as plt

###############################################################################

conv = 666/(2*1.0)

f = r'C:\Users\LakshmanS\Desktop\hahaha\2018.09.09 16-53\Phase\Image'
os.chdir(f)

phase_images = os.listdir(f)

im1 = np.asarray(Image.open(phase_images[0]))

# im1 = np.zeros((900,900))

# for i in range(0,900):
#     for j in range(0,900):
#         #im1[j,i] = np.sin(i/20)
#         im1[j,i] = ((i+j)/(900+900))*255
        #print(i,j)

# plt.imshow(im1,'gray')
# plt.show()
#
# input()
# plt.imshow(im1, 'gray')
# plt.show()

im2 = exposure.rescale_intensity(1.0*im1, in_range=(0, 255), out_range=(0, 4*np.pi))

im3 = np.angle(np.exp(1j * im2))

im4 = restoration.unwrap_phase(im3)

unwrap_per=4*np.pi
ran = np.amax(im4)-np.amin(im4)
nper = np.floor(ran/unwrap_per)
rest = (ran - nper*unwrap_per)/ran

im5 = exposure.rescale_intensity(im4 - np.amin(im4),
                                  in_range=(0, ran),
                                  out_range=(0, conv*(nper + rest)))

###############################################################################

ax1 = plt.subplot(241)
ax1.imshow(im1,'gray')
ax1.set_title('[0, 255]')
plt.xlim(1,900)
plt.ylim(1,900)
plt.xticks([1, 900])
plt.yticks([1, 900])

ax2 = plt.subplot(242)
ax2.imshow(im2,'gray')
ax2.set_title(r'$[0, 4\pi]$')
plt.xlim(1,900)
plt.ylim(1,900)
plt.xticks([1, 900])
plt.yticks([1, 900])

ax3 = plt.subplot(243)
ax3.imshow(im3,'gray')
ax3.set_title(r'$[-\pi, +\pi]$')
plt.xlim(1,900)
plt.ylim(1,900)
plt.xticks([1, 900])
plt.yticks([1, 900])

ax4 = plt.subplot(244)
ax4.imshow(im4,'gray')
ax4.set_title(r'Modified image: ??')
plt.xlim(1,900)
plt.ylim(1,900)
plt.xticks([1, 900])
plt.yticks([1, 900])

ax5 = plt.subplot(245)
ax5.margins(0.05)           # Default margin is 0.05, value 0 means fit
ax5.imshow(im5,'gray')
ax5.set_title(r'Modified image: ??')
plt.xlim(1,900)
plt.ylim(1,900)
plt.xticks([1, 900])
plt.yticks([1, 900])

plt.show()
