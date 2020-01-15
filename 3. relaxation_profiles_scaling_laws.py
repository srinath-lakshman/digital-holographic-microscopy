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
from FUNC_ import sphere_to_cylinder

################################################################################

f = r'/media/devici/Samsung_T5/srinath_dhm/impact_over_thin_films/speed1/00200cs0015mum_r1/2018.09.09 20-47'

avg_back, avg_def, img1, img2, img3, xc, yc, t = read_info_file(f)

conv = 2.967841e-06                                                             #2.5x magnification

################################################################################

os.chdir(f)
os.chdir(os.getcwd() + r'/Unwrapped')

def_files = sorted(glob.glob('h_*.txt'), key=os.path.getmtime)
n = np.shape(def_files)[0]

r = np.loadtxt('r.txt')
t = np.loadtxt('t.txt')

def_img = np.zeros((len(r),len(t)))

h_min = np.zeros(len(t))
h_max = np.zeros(len(t))

r_min = np.zeros(len(t))
r_max = np.zeros(len(t))

count = 0

for i in range(n):
    print(def_files[i])
    def_img[:,i] = np.loadtxt(def_files[i])

    h_min[count] = min(def_img[:,i])
    h_max[count] = max(def_img[:,i])

    r_min[count] = r[int(np.argmin(def_img[:,i]))]
    r_max[count] = r[int(np.argmax(def_img[:,i]))]

    count = count + 1

os.chdir('..')
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')

plt.figure()
plt.plot(range(len(r_min[:])),r_min[:])
plt.scatter(range(len(r_min[:])),r_min[:])
plt.title('r_min')

plt.figure()
plt.plot(range(len(r_max[:])),r_max[:])
plt.scatter(range(len(r_max[:])),r_max[:])
plt.title('r_max')

plt.figure()
plt.plot(range(len(h_min[:])),h_min[:])
plt.scatter(range(len(h_min[:])),h_min[:])
plt.title('h_min')

plt.figure()
plt.plot(range(len(h_max[:])),h_max[:])
plt.scatter(range(len(h_max[:])),h_max[:])
plt.title('h_max')

plt.figure()
plt.plot(r*(10**3),def_img[:,0]*(10**6))
plt.scatter(r*(10**3),def_img[:,0]*(10**6))
plt.scatter(r_min*(10**3), h_min*(10**6), marker='x',c ='black')
plt.scatter(r_max*(10**3), h_max*(10**6), marker='x',c ='black')
plt.xlabel(r'r [mm]')
plt.ylabel(r'$\delta$ [$\mu m$]')
plt.savefig('relaxation_scaling.png')

plt.show()
