import numpy as np
import matplotlib.pyplot as plt
import os

f = r'D:\srinath_dhm\impact_over_thin_films\00050cs0010mum_r1\2018.09.09 17-17'
os.chdir(f)
os.chdir(os.getcwd() + r'\info')

def_profile = np.loadtxt('avg_deformation.txt')

x = def_profile[:,0]
y = def_profile[:,1]

fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.set_title("Plot title")
ax1.set_xlabel('x label')
ax1.set_ylabel('y label')

ax1.plot(x,y, c='r', label='the data')

leg = ax1.legend()

plt.show()
