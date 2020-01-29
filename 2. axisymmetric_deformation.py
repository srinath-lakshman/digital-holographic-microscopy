from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import cv2
import sys
from FUNC_ import read_info_file
from FUNC_ import max_ray_pixels
from FUNC_ import average_profile

f = r'/media/devici/Samsung_T5/srinath_dhm/impact_over_thin_films/speed1/00100cs0010mum_r1/2018.09.09 18-53'

avg_back, avg_def, img1, img2, img3, xc, yc, t = read_info_file(f)

conv = 2.967841e-06                                                             #pixel side length for 2.5x magnification

################################################################################

os.chdir(f)
os.chdir(os.getcwd() + r'/Unwrapped')

def_files = sorted(glob.glob('*.txt'), key=os.path.getmtime)

x_px = np.shape(np.loadtxt(def_files[0]))[0]                                    #x_px = 900
y_px = np.shape(np.loadtxt(def_files[0]))[1]                                    #y_px = 900

s = max_ray_pixels(xc,yc,x_px,y_px)

delta_theta = round(np.degrees(np.arcsin(16/((x_px + y_px)/1))),1)              #delta_theta = 0.5 degrees
n = int(round(360/delta_theta))                                                 #n = 720

# theta = 22.5
# theta_inc = 45 + 22.5
#
# theta_start = theta - theta_inc
# theta_end = theta + theta_inc

b = np.zeros((img3-img2-1,s))
b1 = np.zeros((img3-img2-1,s))

tt = np.zeros(img3-img2-1)

r = np.arange(0,s)*conv

for l in range(0,img3-img2-1):
    i = l + img2+1
    def_img = np.loadtxt(def_files[i])
    print(def_files[i])
    theta_profiles = average_profile(0, 360, delta_theta, xc, yc, s, avg_back-def_img)
    theta_profiles1 = average_profile(-45, 90, delta_theta, xc, yc, s, avg_back-def_img)

    b[l,:] = theta_profiles
    b1[l,:] = theta_profiles1

    tt[l] = float(t[i])-float(t[img1])

    fig1 = plt.figure(1)
    ax1 = plt.gca()
    plt.xlabel(r'$r$ [mm]')
    plt.ylabel(r'$z - h_0$ [nm]')
    plt.title('Number of rays: %i' %n)
    box1 = ax1.get_position()
    ax1.set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])
    ax1.plot(r[:]*(10**(3)),b[l,:]*(10**(0)),label='%i ms' %tt[l])
    ax1.legend(bbox_to_anchor=(1.3, 1))
    plt.grid(True)

    fig2 = plt.figure(2)
    ax2 = plt.gca()
    plt.xlabel(r'$r$ [mm]')
    plt.ylabel(r'$z - h_0$ [nm]')
    plt.title('Number of rays: %i' %n)
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])
    ax2.plot(r[:]*(10**(3)),b1[l,:]*(10**(0)),label='%i ms' %tt[l])
    ax2.legend(bbox_to_anchor=(1.3, 1))
    plt.grid(True)

    ax1.set_xlim([min(ax1.get_xlim()[0],ax2.get_xlim()[0]),max(ax1.get_xlim()[1],ax2.get_xlim()[1])])
    ax1.set_ylim([min(ax1.get_ylim()[0],ax2.get_ylim()[0]),max(ax1.get_ylim()[1],ax2.get_ylim()[1])])

    ax2.set_xlim([min(ax1.get_xlim()[0],ax2.get_xlim()[0]),max(ax1.get_xlim()[1],ax2.get_xlim()[1])])
    ax2.set_ylim([min(ax1.get_ylim()[0],ax2.get_ylim()[0]),max(ax1.get_ylim()[1],ax2.get_ylim()[1])])

os.chdir('..')
os.chdir(os.getcwd() + r'/info')

if not os.path.isdir(os.getcwd() + r'/axisymmetric_short_time'):
     os.makedirs(os.getcwd() + r'/axisymmetric_short_time')
os.chdir(os.getcwd() + r'/axisymmetric_short_time')

np.savetxt('axisymmetric_short_time.txt', b, delimiter='\t')
np.savetxt('semi_axisymmetric_short_time.txt', b1, delimiter='\t')
np.savetxt('time.txt', tt, delimiter='\t')

fig1.savefig('axisymmetric_short_time.png', bbox_inches='tight')
fig2.savefig('semi_axisymmetric.png', bbox_inches='tight')

#saving initial profile
np.savetxt('profile_initial.txt', np.transpose([r[:],b1[0,:]*(10**(-9))]), delimiter='\t')

plt.figure()
plt.plot(r[:],b1[0,:])
plt.show()

print('DONE!!')

################################################################################
