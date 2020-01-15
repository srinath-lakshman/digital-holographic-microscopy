from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import cv2
import sys
from FUNC_ import deformation_center
from FUNC_ import dhm_background_image_analysis
from FUNC_ import read_info_file

################################################################################

f = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00100cs0010mum_r2_TODO/2018.09.09 18-57'

img1 = int('00793')                                                             #1st impact interference
img2 = int('00799')                                                             #1st rebound interference
img3 = int('00812')                                                             #2nd impact interference

conv = 2.967841e-06                                                             #pixel length (in meters) for 2.5x magnification

################################################################################

os.chdir(f)
os.chdir(os.getcwd() + r'/Unwrapped')

image, avg_def, avg_back = dhm_background_image_analysis(f, img1, img2, img3, conv)

os.chdir('..')
if not os.path.isdir(os.getcwd() + r'/info'):
     os.makedirs(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/info')

np.savetxt('avg_deformation.txt', avg_def, fmt='%0.2f', delimiter='\t')
np.savetxt('avg_background.txt', avg_back, fmt='%0.2f', delimiter='\t')

plt.figure()
plt.plot(avg_def[:,0],avg_def[:,1])
plt.axvline(x=img1, linestyle='--', color='black')
plt.axvline(x=img2, linestyle='--', color='black')
plt.axvline(x=img3, linestyle='--', color='black')
plt.savefig('average_deformation.png')

################################################################################

x_max, y_max, xc, yc, delta_theta = deformation_center(image)

# data = np.zeros((n,2))
# data[:,0] = x_max
# data[:,1] = y_max
# np.savetxt('circle_fitting_points.txt', data, delimiter='\t')

circle_pts = int((len(x_max) + len(y_max))/2)

def_circle_pts = np.zeros((circle_pts,2))

def_circle_pts[:,0] = y_max
def_circle_pts[:,1] = x_max

np.savetxt('deformation_center.txt', def_circle_pts, fmt='%d', delimiter='\t')

plt.figure()
plt.title(r'%05i_unwrapped.txt, $\Delta \theta$ = %.2f$^{\circ}$' %(img2+1, delta_theta))
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
plt.savefig('deformation_center.png')

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
