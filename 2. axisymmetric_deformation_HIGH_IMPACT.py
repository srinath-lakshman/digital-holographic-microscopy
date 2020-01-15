from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import cv2
import sys
from FUNC_ import read_info_file
import matplotlib as mpl

f = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00050cs0010mum_r3*/2018.09.09 17-29'

avg_back, avg_def, img1, img2, img3, xc, yc, t = read_info_file(f)

conv = 2.967841e-06                                                             #pixel length (in meters) for 2.5x magnification

theta_start = 90 + 45
theta_end = 270 - 45

################################################################################

os.chdir(f)
os.chdir(os.getcwd() + r'/Unwrapped')

def_files = sorted(glob.glob('*.txt'), key=os.path.getmtime)

x_px = np.shape(np.loadtxt(def_files[0]))[0]
y_px = np.shape(np.loadtxt(def_files[0]))[1]

for jj1 in range(0,int((x_px+y_px)/2)):
    xx = xc+(jj1*np.cos(np.deg2rad(-theta_start)))
    yy = yc+(jj1*np.sin(np.deg2rad(-theta_start)))
    if xx <= 0 or xx >= x_px or yy <= 0 or yy >= y_px:
        break

s1 = jj1

for jj2 in range(0,int((x_px+y_px)/2)):
    xx = xc+(jj2*np.cos(np.deg2rad(-theta_end)))
    yy = yc+(jj2*np.sin(np.deg2rad(-theta_end)))
    if xx <= 0 or xx >= x_px or yy <= 0 or yy >= y_px:
        break

s2 = jj2

s = int(min(s1,s2))-1
delta_theta = round(np.degrees(np.arcsin(16/((x_px + y_px)/1))),1)
n = int(round((theta_end - theta_start)/delta_theta))

ph = np.zeros((n,s))
b = np.zeros((img3-img2-1,s))
tt = np.zeros(img3-img2-1)
r = np.zeros(s)

fig = plt.figure()
ax = plt.subplot(111)

plt.xlabel(r'$r$ [mm]')
plt.ylabel(r'$z - h_0$ [nm]')

# plt.xlim(0,1.5)
# plt.xticks([0, 0.5, 1.0, 1.5])
#
# plt.ylim(-200,+200)
# plt.yticks([-200, -100, 0, +100, +200])

plt.title('Number of rays: %i' %n)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

for l in range(0,img3-img2-1):
    i = l + img2+1
    def_img = np.loadtxt(def_files[i])
    print(def_files[i])
    count = -1
    for k in np.arange(theta_start,theta_end,delta_theta):
        theta = k
        count = count + 1
        for j in range(0,s):
            xx = xc+(j*np.cos(np.deg2rad(-theta)))
            yy = yc+(j*np.sin(np.deg2rad(-theta)))
            ph[count,j] = avg_back[int(round(yy)),int(round(xx))] - def_img[int(round(yy)),int(round(xx))]
            r[j] = j*conv
            b[l,j] = b[l,j] + ph[count,j]
    b[l,:] = (b[l,:]-np.mean(b[l,:]))/n
    tt[l] = float(t[i])-float(t[img1])
    ax.plot(r[:]*(10**(3)),b[l,:]*(10**(0)),label='%i ms' %tt[l])
    ax.legend(bbox_to_anchor=(1.3, 1))

os.chdir('..')
os.chdir(os.getcwd() + r'/info')

if not os.path.isdir(os.getcwd() + r'/axisymmetric_short_time'):
     os.makedirs(os.getcwd() + r'/axisymmetric_short_time')
os.chdir(os.getcwd() + r'/axisymmetric_short_time')

np.savetxt('axisymmetric_short_time.txt', b, delimiter='\t')
np.savetxt('time.txt', tt, delimiter='\t')

plt.savefig('axisymmetric_short_time.png')

print('DONE!!')

################################################################################
