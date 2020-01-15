import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
import matplotlib as mpl

conv = 2.967841e-06     #2.5x magnification

################################################################################

f1 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0005mum_r1_GOOD/2018.09.09 20-17'
os.chdir(f1)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/axisymmetric_short_time')
time_1 = np.loadtxt('time.txt')
b_1 = np.loadtxt('axisymmetric_short_time.txt')
r_1 = np.arange(0,len(b_1[0,:]))*conv

f2 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0005mum_r2/2018.09.09 20-20'
os.chdir(f2)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/axisymmetric_short_time')
time_2 = np.loadtxt('time.txt')
b_2 = np.loadtxt('axisymmetric_short_time.txt')
r_2 = np.arange(0,len(b_2[0,:]))*conv

f3 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0005mum_r3/2018.09.09 20-27'
os.chdir(f3)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/axisymmetric_short_time')
time_3 = np.loadtxt('time.txt')
b_3 = np.loadtxt('axisymmetric_short_time.txt')
r_3 = np.arange(0,len(b_3[0,:]))*conv

################################################################################

min_time = min(min(time_1),min(time_2),min(time_3)).astype(int)
max_time = max(max(time_1),max(time_2),max(time_3)).astype(int)
levels = ((max_time - min_time)/2) + 1

c = np.arange(0, levels)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.copper)
cmap.set_array([])

################################################################################

gs = gridspec.GridSpec(18, 7)

fig = plt.subplot(gs[0:8, 0:3])
ax = plt.gca()

c_1 = np.arange(0, len(time_1))
norm_1 = mpl.colors.Normalize(vmin=c_1.min(), vmax=c_1.max())
cmap_1 = mpl.cm.ScalarMappable(norm=norm_1, cmap=mpl.cm.copper)
cmap_1.set_array([])

plt.xlabel(r'$r$ [mm]')
plt.ylabel(r'$z - h_0$ [nm]')
plt.title('200 mPa.s, 05 $\mu m$, run1')

plt.xlim(0,1.2)
plt.xticks([0, 0.4, 0.8, 1.2])
plt.ylim(-200,+200)
plt.yticks([-200, -100, 0, +100, +200])

plt.grid(True)

for i in np.arange(0,len(b_1[:,0])):
    ax.plot(r_1*(10**3),b_1[i,:], c=cmap_1.to_rgba(i+1))

# cbar_1 = plt.colorbar(cmap_1)
# cbar_1.set_ticks(c_1)
# cbar_1.set_ticklabels(time_1.astype(int))

################################################################################

plt.subplot(gs[0:8, 4:7])
ax = plt.gca()

c_2 = np.arange(0, len(time_2))
norm_2 = mpl.colors.Normalize(vmin=c_2.min(), vmax=c_2.max())
cmap_2 = mpl.cm.ScalarMappable(norm=norm_2, cmap=mpl.cm.copper)
cmap_2.set_array([])

plt.xlabel(r'$r$ [mm]')
plt.ylabel(r'$z - h_0$ [nm]')
plt.title('200 mPa.s, 05 $\mu m$, run2')

plt.xlim(0,1.2)
plt.xticks([0, 0.4, 0.8, 1.2])
plt.ylim(-200,+200)
plt.yticks([-200, -100, 0, +100, +200])

plt.grid(True)

for i in np.arange(0,len(b_2[:,0])):
    ax.plot(r_2*(10**3),b_2[i,:], c=cmap_2.to_rgba(i+1))

# cbar_2 = plt.colorbar(cmap_2)
# cbar_2.set_ticks(c_2)
# cbar_2.set_ticklabels(time_2.astype(int))

################################################################################

plt.subplot(gs[9:16, 2:5])
ax = plt.gca()

c_3 = np.arange(0, len(time_3))
norm_3 = mpl.colors.Normalize(vmin=c_3.min(), vmax=c_3.max())
cmap_3 = mpl.cm.ScalarMappable(norm=norm_3, cmap=mpl.cm.copper)
cmap_3.set_array([])

plt.xlabel(r'$r$ [mm]')
plt.ylabel(r'$z - h_0$ [nm]')
plt.title('200 mPa.s, 05 $\mu m$, run3')

plt.xlim(0,1.2)
plt.xticks([0, 0.4, 0.8, 1.2])
plt.ylim(-200,+200)
plt.yticks([-200, -100, 0, +100, +200])

plt.grid(True)

for i in np.arange(0,len(b_3[:,0])):
    ax.plot(r_3*(10**3),b_3[i,:], c=cmap_3.to_rgba(i+1))

# cbar_3 = plt.colorbar(cmap_3)
# cbar_3.set_ticks(c_3)
# cbar_3.set_ticklabels(time_3.astype(int))

################################################################################

fig_f = plt.subplot(gs[17:18, 0:7])
cbar = plt.colorbar(cmap, cax=fig_f, ticks=[c.min(), c.max()], orientation='horizontal')
cbar.set_ticklabels([min_time, max_time])
cbar.set_label('t [ms]')

plt.show()

################################################################################
