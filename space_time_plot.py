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
from mpl_toolkits import mplot3d
import matplotlib as mpl

###############################################################################

f = r'/media/devici/Samsung_T5/srinath_dhm/impact_over_thin_films/speed1/00100cs0010mum_r1/2018.09.09 18-53'
os.chdir(f)
os.chdir(os.getcwd() + r'/Unwrapped')

def_files = sorted(glob.glob('h_*.txt'), key=os.path.getmtime)
n = np.shape(def_files)[0]

r = np.loadtxt('r.txt')
t = np.loadtxt('t.txt')

lr = len(r)
lt = len(t)

def_prof = np.zeros((lt-1,lr))

# lt = 10
xx = 66
# print(def_files[0])
# input()

R, T = np.meshgrid(r[xx:], t[0:lt-1])

# print(np.shape(R))
# print(np.shape(T))
# print(np.shape(def_prof))
# input()

min_time = t[0]
max_time = t[-1]
levels = ((max_time - min_time)/2) + 1

c = np.arange(0, levels)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.copper)
cmap.set_array([])

cmap = mpl.cm.get_cmap('Spectral')

fig = plt.figure()
ax = plt.axes()

for d in ["left", "top", "bottom", "right"]:
    plt.gca().spines[d].set_visible(False)

# ax[0].spines['right'].set_visible(False)

# plt.axis('off')

# yruler = ax.YRuler;
# yruler.Axle.Visible = 'off';

# ax.tick_params(axis=u'both', which=u'both',length=0)

plt.xlim(0.195,1.4)
plt.xticks([0.2,0.4,0.6,0.8,1])
plt.ylim(-525,3250)
plt.yticks([-500,0,500],[-0.5,0,+0.5])
# plt.hlines(0,0.2,1.0,colors='Gray',linestyles='--', linewidth=0.5)

plt.vlines(0.2,-500,500,colors='Gray',linestyles='-', linewidth=0.5)
plt.hlines(-500,0.2,1.0,colors='Gray',linestyles='-', linewidth=0.5)
# plt.xlim(-1,1)
# plt.ylim(-1,1)
# plt.ylim(0.01,4)

plt.xlabel('r [$mm$]')
plt.ylabel(r'$\delta$ [$\mu m$]')

ax.yaxis.set_label_coords(-0.05,0.125)
ax.xaxis.set_label_coords(+0.35,-0.05)

plt.text(0.2+(-4*0.025)+1,0,'15 ms',fontsize='10')
# plt.text(0.0002+0.000025,0.00000030,'40 ms',fontsize='15')
# plt.text(0.0002+0.000050,0.00000060,'80 ms',fontsize='15')
# plt.text(0.0002+0.000075,0.00000090,'160 ms',fontsize='15')
# plt.text(0.0002+0.000100,0.00000120,'320 ms',fontsize='15')
plt.text(0.2+(4*0.025)+1,(11*250),'1500 ms',fontsize='10')

plt.arrow(0.2+(-3*0.025)+1,(1*250),(7*0.025),(9*250), head_width=0.01, head_length=100, linewidth=0.25, color='Gray', length_includes_head=True)
# plt.plot([0.2,0.2],[500,-500], linewidth=0.5, color="black")
# plt.plot([0.2,1.0],[-500,-500], linewidth=0.5, color="black")

plt.plot([0.2,0.2+0.250],[0,(12*250)], linewidth=0.5, linestyle='--', color="Gray")
plt.plot([1,1+0.250],[0,(12*250)], linewidth=0.5, linestyle='--', color="Gray")

# plt.show()

count = 0

count1 = 0

count2 = 9

# print(lt)
# input()

# aa = [1, 10**(0.5), 10, 100**(0.5), 100, 1000**(0.5), 1000]
# aa = [10, 20, 40, 80, 160, 320, 640]
aa = [0, 10**((1)*(math.log10(890)/9)), 10**((2)*(math.log10(890)/9)), 10**((3)*(math.log10(890)/9)), 10**((4)*(math.log10(890)/9)), 10**((5)*(math.log10(890)/9)), 10**((6)*(math.log10(890)/9)), 10**((7)*(math.log10(890)/9)), 10**((8)*(math.log10(890)/9)), 890]

n = 10
colors = plt.cm.Wistia(np.linspace(0,1,n))

# print(n)
# input()

for i in np.arange(0,lt-1):
    print(def_files[i])
    def_prof[i,:] = np.loadtxt(def_files[i])

    # if count == 0 or count == 125 or count == 250 or count == 375 or count == 500 or count == 625 or count == 750 or count == 875:
    if count in np.around(aa):
        # ax.plot(r,def_prof[count,:])
        ax.plot((r[68:337] + ((count1)*(0.000025/125)))*(10**3),(def_prof[count,68:337]+(count1*2.5*(10**-9)))*(10**9), color=colors[count2])
        # ax.scatter(r[:] + ((count1)*(0.000015/125)),def_prof[count,:]+(count1*2*(10**-9)), color='black')
        # ax.scatter(r[10:1000] + ((count1)*(0.000015/125)),def_prof[count,10:1000]+(count1*2*(10**-9)), color='black')
        # ax.scatter(r[1:] + ((count1)*(0.000015/125)),smooth(def_prof[count,1:],2)+(count1*2*(10**-9)), color='black')
        # ax.plot(r, def_prof[count,:]+(count1*2*(10**-9)))
        # print((count1)*(0.000025/125))
        count1 = count1 + 125
        count2 = count2 - 1
        # print(count*2*(10**-9))
        # ax.plot(r,def_prof[count,:]+(math.log10(count)*1*(10**-9)))
        # ax.scatter(r,t[count],def_prof[count,:],color = '#e69f00')
        # ax.plot(r,def_prof[count,:],t[count])

    # index_min = np.argmin(def_prof[i,:])
    # index_max = np.argmax(def_prof[i,:])
    #
    # h_min = min(smooth_theta_profiles[nn:])
    # h_max = max(smooth_theta_profiles[nn:])
    #
    count = count + 1

# # ax.plot_wireframe(R, T, def_prof, cmap='viridis')
# ax.contour(R*(10**3), T, def_prof[:,xx:]*(10**9), cmap='viridis')
# # ax.contourf(R*(10**3), T, def_prof[:,xx:]*(10**9), cmap='Blues')
# ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.grid(False)
# ax.view_init(azim=-45, elev=+33)
# ax.set_xlabel(r'r [mm]')
# ax.set_xlim(0,1)
# ax.set_ylabel(r't [s]')
# ax.set_ylim(0,2)
# ax.set_zlabel(r'$\delta$ [nm]')
# ax.set_zlim(-300,+300)
# # ax.set_xlim(0.0003,0.001)
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')

# ax.set_yscale('log')

ax.grid(False)

plt.show()
