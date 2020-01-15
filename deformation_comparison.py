import numpy as np
import matplotlib.pyplot as plt
import os

conv = 2.967841e-06     #2.5x magnification

##############################################################################################################################################################
#50 cSt deformation vs height

f1 = r'D:\srinath_dhm\impact_over_thin_films\00050cs0005mum_r1\2018.09.09 16-53'
os.chdir(f1)
os.chdir(os.getcwd() + r'\info')
os.chdir(os.getcwd() + r'\axisymmetric_short_time')
b1 = np.loadtxt('axisymmetric_short_time.txt')
r1 = np.arange(0,np.shape(b1)[1],1).transpose()*conv

f2 = r'D:\srinath_dhm\impact_over_thin_films\00050cs0010mum_r1\2018.09.09 17-17'
os.chdir(f2)
os.chdir(os.getcwd() + r'\info')
os.chdir(os.getcwd() + r'\axisymmetric_short_time')
b2 = np.loadtxt('axisymmetric_short_time.txt')
r2 = np.arange(0,np.shape(b2)[1],1).transpose()*conv

f3 = r'D:\srinath_dhm\impact_over_thin_films\00050cs0015mum_r1\2018.09.09 17-39'
os.chdir(f3)
os.chdir(os.getcwd() + r'\info')
os.chdir(os.getcwd() + r'\axisymmetric_short_time')
b3 = np.loadtxt('axisymmetric_short_time.txt')
r3 = np.arange(0,np.shape(b3)[1],1).transpose()*conv

fig = plt.figure()
ax = plt.subplot(111)

plt.xlim(0,1.2)
plt.ylim(-1200,1200)
plt.xlabel(r'$r$ [mm]')
plt.xticks([0, 0.4, 0.8, 1.2])
plt.ylabel(r'$z - h_0$ [nm]')
plt.yticks([-1200, -800, -400, 0, +400, +800, +1200])

plt.plot(r1*(10**(3)),b1[1]*(10**(0)), label='$h_0 = 5$ $\mu m$')
plt.plot(r2*(10**(3)),b2[1]*(10**(0)), label='$h_0 = 10$ $\mu m$')
plt.plot(r3*(10**(3)),b3[1]*(10**(0)), label='$h_0 = 15$ $\mu m$')
ax.legend()

plt.show()

##############################################################################################################################################################
