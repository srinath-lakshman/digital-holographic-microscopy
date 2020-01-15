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

conv = 2.967841e-06

# #########################################################################
#
# f1 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0015mum_r1_GOOD/2018.09.09 20-47'
# os.chdir(f1)
# os.chdir(os.getcwd() + r'/info')
# os.chdir(os.getcwd() + r'/relaxation_profiles')
# amp_dec = np.loadtxt('amplitude_decay.txt')
# wav_gro = np.loadtxt('wavelength_growth.txt')
# t_amp = amp_dec[:,0]
# A = amp_dec[:,1]
# t_wav = wav_gro[:,0]
# L = wav_gro[:,1]
#
# initial_amp, power_amp, A_fit = power_law_fit(t_amp, A)
# initial_wav, power_wav, L_fit = power_law_fit(t_wav, L)
#
# plt.figure()
# ax = plt.gca()
# # plt.title(r'$\mu$ = %03i mPa s, $h_0$ = %02i $\mu m$, run%i' %(int(f[63:68]), int(f[70:74]), int(f[79])) )
# ax.set_yscale('log')
# ax.set_xscale('log')
# plt.xlim(0.01,10)
# plt.ylim(10,10000)
# plt.xlabel(r'$t$ $[s]$')
# plt.ylabel(r'$A_{\delta}$ $[nm]$')
# ax.scatter(t_amp, A, c='blue', label='Exp')
# ax.plot(t_amp, A_fit, c='red', label=r'$A_{\delta} = $ %i $t^{%0.3f}$' %(initial_amp, power_amp))
# handles,labels = ax.get_legend_handles_labels()
# handles = [handles[1], handles[0]]
# labels = [labels[1], labels[0]]
# ax.legend(handles,labels)
# #ax.grid(True,which='both',ls='-')
# # plt.savefig('amplitude_decay.png')
#
# plt.figure()
# ax = plt.gca()
# # plt.title(r'$\mu$ = %03i mPa s, $h_0$ = %02i $\mu m$, run%i' %(int(f[63:68]), int(f[70:74]), int(f[79])) )
# ax.set_yscale('log')
# ax.set_xscale('log')
# plt.xlim(0.01,10)
# plt.ylim(10,10000)
# plt.xlabel(r'$t$ $[s]$')
# plt.ylabel(r'$L_{\lambda}$ $[\mu m]$')
# ax.scatter(t_wav, L, c='blue', label='Exp')
# ax.plot(t_wav, L_fit, c='red', label=r'$L_{\delta} = $ %i $t^{%0.3f}$' %(initial_wav, power_wav))
# handles,labels = ax.get_legend_handles_labels()
# handles = [handles[1], handles[0]]
# labels = [labels[1], labels[0]]
# ax.legend(handles,labels)
# #ax.grid(True,which='both',ls='-')
# # plt.savefig('wavelength_growth.png')
#
# plt.show()
#
# #########################################################################

##########################################################################

f1 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0005mum_r1/2018.09.09 20-17'
os.chdir(f1)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/axisymmetric_short_time')
b = np.loadtxt('axisymmetric_short_time.txt')
profile_initial = np.loadtxt('profile_intial.txt')

plt.figure()
ax = plt.gca()
# plt.title(r'$\mu$ = %03i mPa s, $h_0$ = %02i $\mu m$, run%i' %(int(f[63:68]), int(f[70:74]), int(f[79])) )
# ax.set_yscale('log')
# ax.set_xscale('log')
# plt.xlim(0.01,10)
# plt.ylim(10,10000)

# ax.plot(b[2,:])
# plt.xlabel(r'$r$ $[m]$')
# plt.ylabel(r'$A_{\delta}$ $[m]$')

ax.plot(profile_initial[:,0]*(10**3),profile_initial[:,1]*(10**9))
plt.xlabel(r'$r$ $[mm]$')
plt.ylabel(r'$A_{\delta}$ $[nm]$')

#ax.grid(True,which='both',ls='-')
# plt.savefig('amplitude_decay.png')

plt.show()

##########################################################################
