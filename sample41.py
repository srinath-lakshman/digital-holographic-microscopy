from matplotlib import pyplot as plt
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

##########################################################################

f1 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0015mum_r1_GOOD/2018.09.09 20-47/info/axisymmetric_short_time'
os.chdir(f1)

b1 = np.loadtxt('semi_axisymmetric_short_time.txt')

r = np.arange(0,len(b1[0,:]))*conv

# print(a.shape)
# input()

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

ax.plot(b1[1,:])
np.savetxt('profile_initial.txt', np.transpose([r[:],b1[1,:]*(10**(-9))]), delimiter='\t')
# plt.xlabel(r'$r$ $[mm]$')
# plt.ylabel(r'$A_{\delta}$ $[nm]$')

#ax.grid(True,which='both',ls='-')
# plt.savefig('amplitude_decay.png')

plt.show()

##########################################################################
