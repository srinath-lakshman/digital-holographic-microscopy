from matplotlib import pyplot as plt
import numpy as np
import os
import glob
from FUNC_ import read_info_file
from FUNC_ import smooth
from FUNC_ import power_law_fit
from FUNC_ import average_profile
from scipy.interpolate import UnivariateSpline
from scipy import stats
import math

##########################################################################

f = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed2/00050cs0010mum_r1/2018.09.10 16-29'

avg_back, avg_def, img1, img2, img3, xc, yc, t = read_info_file(f)

conv = 2.967841e-06                                                             #2.5x magnification

##########################################################################

os.chdir(f)
os.chdir(os.getcwd() + r'/Unwrapped')

def_files = sorted(glob.glob('*.txt'), key=os.path.getmtime)
n = np.shape(def_files)[0]                                                      #n = 2000

x_px = np.shape(np.loadtxt(def_files[0]))[0]                                    #x_px = 900
y_px = np.shape(np.loadtxt(def_files[0]))[1]                                    #y_px = 900

s = min(xc,yc,x_px-xc-1,y_px-yc-1)-1
s = 600

def_second = np.loadtxt(def_files[img3+6])
theta = 180                                                         #central angle
theta_inc = 22.5                                                           #max increment in central angle
# theta_inc = 1                                                                 #max increment in central angle
delta_theta = 0.5                                                               #increment in angle
p = 10                                                                        #skip files
smth = 20                                                                       #smooth parameter
nn = 225                                                                        #ignore profile for index value below

kk = np.shape(np.arange(img2+1,n,p))[0]

index_max = np.zeros(kk)
index_min = np.zeros(kk)

h_min = np.zeros(kk)
h_max = np.zeros(kk)

time = np.zeros(kk)

for i in np.arange(img2+1,n,p):
    count = int(round(((i-img2-2)/p)))
    print(def_files[i])
    def_img = np.loadtxt(def_files[i])
    time[count] = (float(t[i]) - float(t[img1-1]))/1000
    theta_profiles = average_profile(theta-theta_inc, theta+theta_inc, delta_theta, xc, yc, s, avg_back-def_img)

    smooth_theta_profiles = smooth(theta_profiles,smth)
    # smooth_theta_profiles = theta_profiles

    index_min[count] = np.argmin(smooth_theta_profiles[nn:]) + nn
    index_max[count] = np.argmax(smooth_theta_profiles[nn:]) + nn

    h_min[count] = min(smooth_theta_profiles[nn:])
    h_max[count] = max(smooth_theta_profiles[nn:])

    # plt.figure()
    # plt.title('deformation profiles; %s; t = %.2f s' %(def_files[i],time[count]))
    # plt.xlim(1,400)
    # plt.ylim(-1200, 1200)
    # plt.xlabel(r'$r$ [mm]')
    # plt.ylabel(r'$z - h_0$ [nm]')
    # # plt.xticks([1, 256])
    # # plt.yticks([1, xnew_px*ynew_px])
    # plt.plot(theta_profiles, c='#ff7f0e')
    # plt.plot(smooth_theta_profiles, c='#1f77b4')
    # plt.scatter(index_min, h_min, marker='x', c='black')
    # plt.scatter(index_max, h_max, marker='x', c='black')
    # # plt.show()
    # plt.draw()
    # plt.savefig('smooth_{0:05d}.png'.format(i))
    # plt.clf()

A = abs(h_max) + abs(h_min)
# A = abs(h_min)
L = abs(abs(index_min) - abs(index_max))*conv*np.power(10,6)

initial_amp, power_amp, A_fit = power_law_fit(time, A)
initial_wav, power_wav, L_fit = power_law_fit(time, L)

amp_dec = np.zeros((kk,2))
wav_gro = np.zeros((kk,2))
detail_txt = np.zeros(6)

amp_dec[:,0] = time[:]
amp_dec[:,1] = A[:]

wav_gro[:,0] = time[:]
wav_gro[:,1] = L[:]

detail_txt[0] = theta
detail_txt[1] = theta_inc
detail_txt[2] = delta_theta
detail_txt[3] = p
detail_txt[4] = smth
detail_txt[5] = nn

os.chdir('..')
os.chdir(os.getcwd() + r'/info')

if not os.path.isdir(os.getcwd() + r'/relaxation_profiles'):
     os.makedirs(os.getcwd() + r'/relaxation_profiles')
os.chdir(os.getcwd() + r'/relaxation_profiles')

np.savetxt('amplitude_decay.txt', amp_dec, delimiter='\t')
np.savetxt('wavelength_growth.txt', wav_gro, delimiter='\t')
np.savetxt('details.txt', detail_txt, delimiter='\t')

plt.figure()
plt.title(r'%s; $\Delta \theta$ = %0.1f$^{\circ}$' %(def_files[img3+6],delta_theta))
plt.xlim(0,900)
plt.ylim(0,900)
plt.xticks([0, 150, 300, 450, 600, 750, 900])
plt.yticks([0, 150, 300, 450, 600, 750, 900])
plt.xlabel('x [px]')
plt.ylabel('y [px]')
plt.imshow(def_second,'gray')
theta1 = theta - (theta_inc)
xs1 = int(round(xc-(s*np.cos(np.deg2rad(-theta1)))))
ys1 = int(round(yc-(s*np.sin(np.deg2rad(-theta1)))))
xm1 = int(round(xc+(nn*np.cos(np.deg2rad(-theta1)))))
ym1 = int(round(yc+(nn*np.sin(np.deg2rad(-theta1)))))
xf1 = int(round(xc+(s*np.cos(np.deg2rad(-theta1)))))
yf1 = int(round(yc+(s*np.sin(np.deg2rad(-theta1)))))
theta2 = theta
xs2 = int(round(xc-(s*np.cos(np.deg2rad(-theta2)))))
ys2 = int(round(yc-(s*np.sin(np.deg2rad(-theta2)))))
xm2 = int(round(xc+(nn*np.cos(np.deg2rad(-theta2)))))
ym2 = int(round(yc+(nn*np.sin(np.deg2rad(-theta2)))))
xf2 = int(round(xc+(s*np.cos(np.deg2rad(-theta2)))))
yf2 = int(round(yc+(s*np.sin(np.deg2rad(-theta2)))))
theta3 = theta + (theta_inc)
xs3 = int(round(xc-(s*np.cos(np.deg2rad(-theta3)))))
ys3 = int(round(yc-(s*np.sin(np.deg2rad(-theta3)))))
xm3 = int(round(xc+(nn*np.cos(np.deg2rad(-theta3)))))
ym3 = int(round(yc+(nn*np.sin(np.deg2rad(-theta3)))))
xf3 = int(round(xc+(s*np.cos(np.deg2rad(-theta3)))))
yf3 = int(round(yc+(s*np.sin(np.deg2rad(-theta3)))))
plt.plot([xc, xm1],[yc, ym1], '--', c='#1f77b4')
plt.plot([xm1, xf1],[ym1, yf1], c='#1f77b4', label=r'$\theta = $ %5.1f $^{\circ}$' %theta1)
plt.plot([xc, xm2],[yc, ym2], '--', c='#ff7f0e')
plt.plot([xm2, xf2],[ym2, yf2], c='#ff7f0e', label=r'$\theta = $ %5.1f $^{\circ}$' %theta2)
plt.plot([xc, xm3],[yc, ym3], '--', c='#2ca02c')
plt.plot([xm3, xf3],[ym3, yf3], c='#2ca02c', label=r'$\theta = $ %5.1f $^{\circ}$' %theta3)
plt.scatter(xc,yc, marker='x', c='black')
plt.gca().invert_yaxis()
plt.legend()
plt.savefig('ray_scanning.png')

plt.figure()
ax = plt.gca()
# plt.title(r'$\mu$ = %03i mPa s, $h_0$ = %02i $\mu m$, run%i' %(int(f[63:68]), int(f[70:74]), int(f[79])) )
ax.set_yscale('log')
ax.set_xscale('log')
# plt.xlim(0.01,10)
# plt.ylim(1,100)
plt.xlabel(r'$t$ $[s]$')
plt.ylabel(r'$A_{\delta}$ $[nm]$')
ax.scatter(time, A, c='blue', label='Exp')
ax.plot(time, A_fit, c='red', label=r'$A_{\delta} = $ %i $t^{%0.3f}$' %(initial_amp, power_amp))
handles,labels = ax.get_legend_handles_labels()
handles = [handles[1], handles[0]]
labels = [labels[1], labels[0]]
ax.legend(handles,labels)
#ax.grid(True,which='both',ls='-')
plt.savefig('amplitude_decay.png')

plt.figure()
ax = plt.gca()
# plt.title(r'$\mu$ = %03i mPa s, $h_0$ = %02i $\mu m$, run%i' %(int(f[63:68]), int(f[70:74]), int(f[79])) )
ax.set_yscale('log')
ax.set_xscale('log')
# plt.xlim(0.01,10)
# plt.ylim(10,1000)
plt.xlabel(r'$t$ $[s]$')
plt.ylabel(r'$L_{\lambda}$ $[\mu m]$')
ax.scatter(time, L, c='blue', label='Exp')
ax.plot(time, L_fit, c='red', label=r'$A_{\delta} = $ %i $t^{%0.3f}$' %(initial_wav, power_wav))
handles,labels = ax.get_legend_handles_labels()
handles = [handles[1], handles[0]]
labels = [labels[1], labels[0]]
ax.legend(handles,labels)
#ax.grid(True,which='both',ls='-')
plt.savefig('wavelength_growth.png')

print('DONE!!')
