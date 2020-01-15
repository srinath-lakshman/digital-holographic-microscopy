import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
import matplotlib as mpl

conv = 2.967841e-06     #2.5x magnification

################################################################################

f1 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00050cs0010mum_r1/2018.09.09 17-17'
os.chdir(f1)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/axisymmetric_short_time')
time_1 = np.loadtxt('time.txt')
b_1 = np.loadtxt('axisymmetric_short_time.txt')
r_1 = np.arange(0,len(b_1[0,:]))*conv

f2 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed2/00050cs0010mum_r2_BAD/2018.09.10 16-35'
os.chdir(f2)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/axisymmetric_short_time')
time_2 = np.loadtxt('time.txt')
b_2 = np.loadtxt('axisymmetric_short_time.txt')
r_2 = np.arange(0,len(b_2[0,:]))*conv

f3 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00100cs0010mum_r1/2018.09.09 18-53'
os.chdir(f3)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/axisymmetric_short_time')
time_3 = np.loadtxt('time.txt')
b_3 = np.loadtxt('axisymmetric_short_time.txt')
r_3 = np.arange(0,len(b_3[0,:]))*conv

f4 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed2/00100cs0010mum_r1/2018.09.10 17-43'
os.chdir(f4)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/axisymmetric_short_time')
time_4 = np.loadtxt('time.txt')
b_4 = np.loadtxt('axisymmetric_short_time.txt')
r_4 = np.arange(0,len(b_4[0,:]))*conv

f5 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0010mum_r1_GOOD/2018.09.09 20-34'
os.chdir(f5)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/axisymmetric_short_time')
time_5 = np.loadtxt('time.txt')
b_5 = np.loadtxt('axisymmetric_short_time.txt')
r_5 = np.arange(0,len(b_5[0,:]))*conv

f6 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed2/00200cs0010mum_r1/2018.09.10 20-12'
os.chdir(f6)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/axisymmetric_short_time')
time_6 = np.loadtxt('time.txt')
b_6 = np.loadtxt('axisymmetric_short_time.txt')
r_6 = np.arange(0,len(b_6[0,:]))*conv

################################################################################

min_time = min(min(time_1),min(time_2),min(time_3),min(time_4),min(time_5),min(time_6)).astype(int)
max_time = max(max(time_1),max(time_2),max(time_3),max(time_4),max(time_5),max(time_6)).astype(int)
levels = ((max_time - min_time)/2) + 1

c = np.arange(0, levels)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.copper)
cmap.set_array([])

# plt.plot(b_2[0,220:520])
# plt.show()

################################################################################

gs = gridspec.GridSpec(18, 7)

xx1_speed1 = 0.63
xx2_speed1 = 0.84

xx1_speed2 = 0.63
xx2_speed2 = 0.96

################################################################################

plt.subplot(gs[0:4, 0:3])
ax = plt.gca()

# c_1 = np.arange(0, len(time_1))
# norm_1 = mpl.colors.Normalize(vmin=c_1.min(), vmax=c_1.max())
# cmap_1 = mpl.cm.ScalarMappable(norm=norm_1, cmap=mpl.cm.copper)
# cmap_1.set_array([])

plt.xlabel(r'$r$ [$mm$]')
plt.ylabel(r'$z - h_0$ [$\mu m$]')
plt.title('050 mPa.s, 10 $\mu m$, 0.13 m/s')

plt.xlim(0,1.7)
plt.xticks([0, xx1_speed1, xx2_speed1, 1.7])
plt.ylim(-0.5,+0.5)
plt.yticks([-0.5, 0, +0.5])

plt.axvline(x=xx1_speed1, color='k', linestyle='--')
plt.axvline(x=xx2_speed1, color='k', linestyle='--')

plt.grid(True)

for i in np.arange(0,len(b_1[:,0])):
    ax.plot(r_1*(10**3),b_1[i,:]/1000, c=cmap.to_rgba(i+1))

# cbar_1 = plt.colorbar(cmap_1)
# cbar_1.set_ticks(c_1)
# cbar_1.set_ticklabels(time_1.astype(int))

################################################################################

plt.subplot(gs[0:4, 4:7])
ax = plt.gca()

# c_2 = np.arange(0, len(time_2))
# norm_2 = mpl.colors.Normalize(vmin=c_2.min(), vmax=c_2.max())
# cmap_2 = mpl.cm.ScalarMappable(norm=norm_2, cmap=mpl.cm.copper)
# cmap_2.set_array([])

plt.xlabel(r'$r$ [$mm$]')
plt.ylabel(r'$z - h_0$ [$\mu m$]')
plt.title('050 mPa.s, 10 $\mu m$, 0.50 m/s')

plt.xlim(0,1.7)
plt.xticks([0, xx1_speed2, xx2_speed2, 1.7])
plt.ylim(-0.5,+0.5)
plt.yticks([-0.5, 0, +0.5])

plt.axvline(x=xx1_speed2, color='k', linestyle='--')
plt.axvline(x=xx2_speed2, color='k', linestyle='--')

plt.grid(True)

for i in np.arange(0,len(b_2[:,0])):
    ax.plot(r_2*(10**3),b_2[i,:]/1000, c=cmap.to_rgba(i+1))

# cbar_2 = plt.colorbar(cmap_2)
# cbar_2.set_ticks(c_2)
# cbar_2.set_ticklabels(time_2.astype(int))

################################################################################

plt.subplot(gs[6:10, 0:3])
ax = plt.gca()

# c_1 = np.arange(0, len(time_3))
# norm_1 = mpl.colors.Normalize(vmin=c_1.min(), vmax=c_1.max())
# cmap_1 = mpl.cm.ScalarMappable(norm=norm_1, cmap=mpl.cm.copper)
# cmap_1.set_array([])

plt.xlabel(r'$r$ [$mm$]')
plt.ylabel(r'$z - h_0$ [$\mu m$]')
plt.title('100 mPa.s, 10 $\mu m$, 0.13 m/s')

plt.xlim(0,1.7)
plt.xticks([0, xx1_speed1, xx2_speed1, 1.7])
plt.ylim(-0.5,+0.5)
plt.yticks([-0.5, 0, +0.5])

plt.axvline(x=xx1_speed1, color='k', linestyle='--')
plt.axvline(x=xx2_speed1, color='k', linestyle='--')

plt.grid(True)

for i in np.arange(0,len(b_3[:,0])):
    ax.plot(r_3*(10**3),b_3[i,:]/1000, c=cmap.to_rgba(i+1))

# cbar_1 = plt.colorbar(cmap_1)
# cbar_1.set_ticks(c_1)
# cbar_1.set_ticklabels(time_1.astype(int))

################################################################################

plt.subplot(gs[6:10, 4:7])
ax = plt.gca()

# c_2 = np.arange(0, len(time_4))
# norm_2 = mpl.colors.Normalize(vmin=c_2.min(), vmax=c_2.max())
# cmap_2 = mpl.cm.ScalarMappable(norm=norm_2, cmap=mpl.cm.copper)
# cmap_2.set_array([])

plt.xlabel(r'$r$ [$mm$]')
plt.ylabel(r'$z - h_0$ [$\mu m$]')
plt.title('100 mPa.s, 10 $\mu m$, 0.50 m/s')

plt.xlim(0,1.7)
plt.xticks([0, xx1_speed2, xx2_speed2, 1.7])
plt.ylim(-0.5,+0.5)
plt.yticks([-0.5, 0, +0.5])

plt.axvline(x=xx1_speed2, color='k', linestyle='--')
plt.axvline(x=xx2_speed2, color='k', linestyle='--')

plt.grid(True)

for i in np.arange(0,len(b_4[:,0])):
    ax.plot(r_4*(10**3),b_4[i,:]/1000, c=cmap.to_rgba(i+1))

# cbar_2 = plt.colorbar(cmap_2)
# cbar_2.set_ticks(c_2)
# cbar_2.set_ticklabels(time_2.astype(int))

################################################################################

plt.subplot(gs[12:16, 0:3])
ax = plt.gca()

# c_1 = np.arange(0, len(time_1))
# norm_1 = mpl.colors.Normalize(vmin=c_1.min(), vmax=c_1.max())
# cmap_1 = mpl.cm.ScalarMappable(norm=norm_1, cmap=mpl.cm.copper)
# cmap_1.set_array([])

plt.xlabel(r'$r$ [$mm$]')
plt.ylabel(r'$z - h_0$ [$\mu m$]')
plt.title('200 mPa.s, 10 $\mu m$, 0.13 m/s')

plt.xlim(0,1.7)
plt.xticks([0, xx1_speed1, xx2_speed1, 1.7])
plt.ylim(-0.5,+0.5)
plt.yticks([-0.5, 0, +0.5])

plt.axvline(x=xx1_speed1, color='k', linestyle='--')
plt.axvline(x=xx2_speed1, color='k', linestyle='--')

plt.grid(True)

for i in np.arange(0,len(b_5[:,0])):
    ax.plot(r_5*(10**3),b_5[i,:]/1000, c=cmap.to_rgba(i+1))

# cbar_1 = plt.colorbar(cmap_1)
# cbar_1.set_ticks(c_1)
# cbar_1.set_ticklabels(time_1.astype(int))

################################################################################

plt.subplot(gs[12:16, 4:7])
ax = plt.gca()

# c_2 = np.arange(0, len(time_2))
# norm_2 = mpl.colors.Normalize(vmin=c_2.min(), vmax=c_2.max())
# cmap_2 = mpl.cm.ScalarMappable(norm=norm_2, cmap=mpl.cm.copper)
# cmap_2.set_array([])

plt.xlabel(r'$r$ [$mm$]')
plt.ylabel(r'$z - h_0$ [$\mu m$]')
plt.title('200 mPa.s, 10 $\mu m$, 0.50 m/s')

plt.xlim(0,1.7)
plt.xticks([0, xx1_speed2, xx2_speed2, 1.7])
plt.ylim(-0.5,+0.5)
plt.yticks([-0.5, 0, +0.5])

plt.axvline(x=xx1_speed2, color='k', linestyle='--')
plt.axvline(x=xx2_speed2, color='k', linestyle='--')

plt.grid(True)

for i in np.arange(0,len(b_6[:,0])):
    ax.plot(r_6*(10**3),b_6[i,:]/1000, c=cmap.to_rgba(i+1))

# cbar_2 = plt.colorbar(cmap_2)
# cbar_2.set_ticks(c_2)
# cbar_2.set_ticklabels(time_2.astype(int))

################################################################################

fig_f = plt.subplot(gs[17:18, 2:5])
cbar = plt.colorbar(cmap, cax=fig_f, ticks=[c.min(), c.max()], orientation='horizontal')
cbar.set_ticklabels([min_time, max_time])
cbar.set_label('t [ms]')

plt.show()

################################################################################
