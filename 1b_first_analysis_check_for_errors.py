from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import cv2
import sys
from FUNC_ import dhm_background_image_analysis
from FUNC_ import read_info_file
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

f = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0010mum_r2/2018.09.09 20-38'

avg_back, avg_def, img1, img2, img3, xc, yc, t = read_info_file(f)

os.chdir(f)
os.chdir(os.getcwd() + r'/Unwrapped')

def_files = sorted(glob.glob('*.txt'), key=os.path.getmtime)

################################################################################

kk = img2+1
delta_theta = 0.5
image = avg_back - np.loadtxt(def_files[kk])

################################################################################

os.chdir('..')
os.chdir(os.getcwd() + r'/info')

def_circle_pts = np.loadtxt('deformation_center_final.txt')
y_max = def_circle_pts[:,0]
x_max = def_circle_pts[:,1]

xc = (max(x_max) + min(x_max))/2
yc = (max(y_max) + min(y_max))/2

print(xc, yc)

plt.figure()
plt.title(r'%05i_unwrapped.txt, $\Delta \theta$ = %.2f$^{\circ}$' %(kk, delta_theta))
plt.imshow(image,'gray')
plt.scatter(x_max,y_max)
plt.scatter(xc,yc)
plt.xlim(0,900)
plt.ylim(0,900)
plt.xticks([0, 150, 300, 450, 600, 750, 900])
plt.yticks([0, 150, 300, 450, 600, 750, 900])
plt.gca().invert_yaxis()
plt.xlabel('x [px]')
plt.ylabel('y [px]')
plt.show()
plt.savefig('deformation_center_final.png')

file_write = open('details.txt','w')
file_write.write('%s\n' % f[14:len(f)])
file_write.write('%05i\n' % img1)
file_write.write('%05i\n' % img2)
file_write.write('%05i\n' % img3)
file_write.write('%i\n' % xc)
file_write.write('%i\n' % yc)
file_write.close()

print('DONE!!')

################################################################################

################################################################################

# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# # Make data.
# X = np.arange(0, 900, 1)
# Y = np.arange(0, 900, 1)
# X, Y = np.meshgrid(X, Y)
# Z = image
#
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
# # Customize the z axis.
# # ax.set_zlim(-1.01, 1.01)
# # ax.zaxis.set_major_locator(LinearLocator(10))
# # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
#
# plt.show()

################################################################################
