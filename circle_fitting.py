import numpy as np
import matplotlib.pyplot as plt
import os
from math import sqrt
from scipy import optimize

##########################################################################

f = r'C:\Users\LakshmanS\Documents\python\dhm'
os.chdir(f)

data = np.loadtxt('circle_fitting_points.txt')

x = data[:,0]
y = data[:,1]

fig = plt.figure()

ax1 = fig.add_subplot(111)

# img2 = 192
# delta_theta = 0.5
#plt.title(r'%05i_unwrapped.txt, $\Delta \theta$ = %.1f $^{\circ}$' %(img2+1, delta_theta))

ax1.set_title("Plot title")
ax1.set_xlabel('x label')
ax1.set_ylabel('y label')

ax1.plot(x,y, c='r', label='the data')

leg = ax1.legend()

plt.show()

##########################################################################

method_2 = "leastsq"

def calc_R(xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return sqrt((x-xc)**2 + (y-yc)**2)

def f_2(c):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(*c)
    return Ri - Ri.mean()

x_m = x.mean()
y_m = y.mean()

center_estimate = x_m, y_m
center_2, ier = optimize.leastsq(f_2, center_estimate)

xc_2, yc_2 = center_2
Ri_2       = calc_R(*center_2)
R_2        = Ri_2.mean()
residu_2   = sum((Ri_2 - R_2)**2)
