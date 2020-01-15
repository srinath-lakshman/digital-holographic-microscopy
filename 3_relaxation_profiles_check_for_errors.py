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

f = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/00050cs0005mum_r3/2018.09.09 17-07/info/relaxation_profiles'

os.chdir(f)

amp_dec = np.loadtxt('amplitude_decay.txt')
wav_gro = np.loadtxt('wavelength_growth.txt')

time = amp_dec[:,0]
A = amp_dec[:,1]

plt.scatter(time, A)
plt.show()
# input()

##########################################################################
