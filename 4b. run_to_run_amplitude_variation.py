import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from FUNC_ import power_law_fit

conv = 2.967841e-06     #2.5x magnification

################################################################################

f1 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0010mum_r1_GOOD/2018.09.09 20-34'
os.chdir(f1)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec1 = np.loadtxt('amplitude_decay.txt')
wav_gro1 = np.loadtxt('wavelength_growth.txt')
t_amp1 = amp_dec1[:,0]
A1 = amp_dec1[:,1]
initial_amp1, power_amp1, A_fit1 = power_law_fit(t_amp1, A1)

f2 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0010mum_r2_TODO/2018.09.09 20-38'
os.chdir(f2)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec2 = np.loadtxt('amplitude_decay.txt')
wav_gro2 = np.loadtxt('wavelength_growth.txt')
t_amp2 = amp_dec2[:,0]
A2 = amp_dec2[:,1]
initial_amp2, power_amp2, A_fit2 = power_law_fit(t_amp2, A2)

f3 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0010mum_r3_TODO/2018.09.09 20-40'
os.chdir(f3)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec3 = np.loadtxt('amplitude_decay.txt')
wav_gro3 = np.loadtxt('wavelength_growth.txt')
t_amp3 = amp_dec3[:,0]
A3 = amp_dec3[:,1]
initial_amp3, power_amp3, A_fit3 = power_law_fit(t_amp3, A3)

################################################################################

gs = gridspec.GridSpec(2, 7)

plt.subplot(gs[0, 0:3])
ax = plt.gca()

ax.set_yscale('log')
ax.set_xscale('log')
plt.xlim(0.01,10)
plt.ylim(10,10000)
plt.xlabel(r't [s]')
plt.ylabel(r'$A_{\delta}$ [nm]')
plt.title('200 mPa.s, 10 $\mu m$, run1')
ax.plot(t_amp1, A_fit1, c='black', label=r'$A_{\delta} = $ %i $t^{%0.3f}$' %(initial_amp1, power_amp1))
ax.scatter(t_amp1, A1, c='sandybrown', label='Exp')
ax.legend(loc=1)

plt.subplot(gs[0, 4:7])
ax = plt.gca()

ax.set_yscale('log')
ax.set_xscale('log')
plt.xlim(0.01,10)
plt.ylim(10,10000)
plt.xlabel(r't [s]')
plt.ylabel(r'$A_{\delta}$ [nm]')
plt.title('200 mPa.s, 10 $\mu m$, run2')
ax.plot(t_amp2, A_fit2, c='black', label=r'$A_{\delta} = $ %i $t^{%0.3f}$' %(initial_amp2, power_amp2))
ax.scatter(t_amp2, A2, c='sandybrown', label='Exp')
ax.legend(loc=1)

plt.subplot(gs[1, 2:5])
ax = plt.gca()

ax.set_yscale('log')
ax.set_xscale('log')
plt.xlim(0.01,10)
plt.ylim(10,10000)
plt.xlabel(r't [s]')
plt.ylabel(r'$A_{\delta}$ [nm]')
plt.title('200 mPa.s, 10 $\mu m$, run3')
ax.plot(t_amp3, A_fit3, c='black', label=r'$A_{\delta} = $ %i $t^{%0.3f}$' %(initial_amp3, power_amp3))
ax.scatter(t_amp3, A3, c='sandybrown', label='Exp')
ax.legend(loc=1)

plt.show()
