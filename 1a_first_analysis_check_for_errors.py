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

################################################################################

def deformation_center_new(image, x_c, y_c, s1):

    x_px = np.shape(image)[0]
    y_px = np.shape(image)[1]

    s = int(round((x_px + y_px)/4))                                             #s = 450

    delta_theta = round(np.degrees(np.arcsin(4/s)),1)                           #delta_theta = 0.5
    n = int(round(360/delta_theta))                                             #n = 720

    # ph = np.zeros((n,len(range(min,max))+0))
    ph = np.zeros((n,s1))
    s_max = np.zeros(n)
    x_max = np.zeros(n)
    y_max = np.zeros(n)

    for k in range(0,n):
        theta = k*(360/n)
        for j in range(0,s1):
            xx = x_c+(j*np.cos(np.deg2rad(theta)))
            yy = y_c+(j*np.sin(np.deg2rad(theta)))
            # print(count)
            ph[k,j] = image[int(round(yy)),int(round(xx))]

        s_max[k] = np.argmax(ph[k,:])
        x_max[k] = x_c + (s_max[k]*np.cos(np.deg2rad(theta)))
        y_max[k] = y_c + (s_max[k]*np.sin(np.deg2rad(theta)))

    xc = (max(x_max) + min(x_max))/2
    yc = (max(y_max) + min(y_max))/2

    return x_max, y_max, xc, yc, delta_theta

################################################################################

f = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0005mum_r3/2018.09.09 20-27'

avg_back, avg_def, img1, img2, img3, xc, yc, t = read_info_file(f)

os.chdir(f)
os.chdir(os.getcwd() + r'/Unwrapped')

def_files = sorted(glob.glob('*.txt'), key=os.path.getmtime)

################################################################################)

kk = img2-1
image = avg_back - np.loadtxt(def_files[kk])

plt.figure()
plt.imshow(image,'gray')
plt.scatter(414,519)
plt.show()

# x_max, y_max, xc, yc, delta_theta = deformation_center_new(image, 425, 425, 425)
x_max, y_max, xc, yc, delta_theta = deformation_center_new(image, 430, 560, 340)

circle_pts = int((len(x_max) + len(y_max))/2)
def_circle_pts = np.zeros((circle_pts,2))

def_circle_pts[:,0] = y_max
def_circle_pts[:,1] = x_max

################################################################################

os.chdir('..')
os.chdir(os.getcwd() + r'/info')

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
# plt.show()
plt.savefig('deformation_center_final.png')

np.savetxt('deformation_center_final.txt', def_circle_pts, fmt='%d', delimiter='\t')

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
