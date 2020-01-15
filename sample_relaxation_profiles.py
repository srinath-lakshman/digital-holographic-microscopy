import numpy as np
import os
from skimage import io, restoration, exposure
from PIL import Image
from matplotlib import pyplot as plt
from FUNC_ import power_law_fit

###############################################################################

f = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0015mum_r1_GOOD/2018.09.09 20-47'
os.chdir(f)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')

amp_dec = np.loadtxt('amplitude_decay.txt')
wav_gro = np.loadtxt('wavelength_growth.txt')

t_amp = amp_dec[:,0]
A = amp_dec[:,1]
t_wav = wav_gro[:,0]
L = wav_gro[:,1]

initial_amp, power_amp, A_fit = power_law_fit(t_amp, A)
initial_wav, power_wav, L_fit = power_law_fit(t_wav, L)

mu = 50*(10**-3)
h0 = 5*(10**-6)

plt.figure()

# plt.scatter(t_wav,L)
plt.plot(L)
plt.show()
