import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
from FUNC_ import power_law_fit
from FUNC_ import sphere_to_pancake
from scipy import stats

################################################################################

conv = 2.967841e-06
sigma = 20.3/1000
nnn = 100

main_path = r'/media/devici/Samsung_T5/srinath_dhm/impact_over_thin_films/speed1/'

################################################################################

f_a = main_path + r'00050cs0005mum_r1/2018.09.09 16-53'; ka = -1
f_b = main_path + r'00050cs0005mum_r2/2018.09.09 16-59'; kb = -1
f_c = main_path + r'00050cs0005mum_r3/2018.09.09 17-07'; kc = -1

mu = 50*(10**-3)
h0 = 5*(10**-6)

# f_a = main_path + r'00050cs0010mum_r1/2018.09.09 17-17'; ka = -1
# f_b = main_path + r'00050cs0010mum_r2/2018.09.09 17-23'; kb = 650
# f_c = main_path + r'00050cs0010mum_r3/2018.09.09 17-29'; kc = -1
#
# mu = 50*(10**-3)
# h0 = 10*(10**-6)

# f_a = main_path + r'00050cs0015mum_r1/2018.09.09 17-39'; ka = 447
# f_b = main_path + r'00050cs0015mum_r2/2018.09.09 17-45'; kb = 482
# f_c = main_path + r'00050cs0015mum_r3/2018.09.09 17-51'; kc = 468
#
# mu = 50*(10**-3)
# h0 = 15*(10**-6)

################################################################################

if h0 == 5*(10**-6):
    mark = "v"
elif h0 == 10*(10**-6):
    mark = "o"
else:
    mark = "^"

if mu == 50*(10**-3):
    col = "#1f77b4"
elif mu == 100*(10**-3):
    col = "#ff7f0e"
else:
    col = "#2ca02c"

################################################################################

os.chdir(f_a)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec_a = np.loadtxt('amplitude_decay.txt')
wav_gro_a = np.loadtxt('wavelength_growth.txt')
t_amp_a = amp_dec_a[0:ka,0]
A_a = amp_dec_a[0:ka,1]
t_wav_a = wav_gro_a[0:ka,0]
L_a = wav_gro_a[0:ka,1]
initial_amp_a, power_amp_a, A_fit_a = power_law_fit(t_amp_a, A_a)
initial_wav_a, power_wav_a, L_fit_a = power_law_fit(t_wav_a, L_a)

L_a_mod, bin_edges_L_a_mod, *_ = stats.binned_statistic(t_wav_a, L_a, 'mean', bins=nnn)
L_a_mod_err, *_ = stats.binned_statistic(t_wav_a, L_a, 'std', bins=nnn)
t_wav_a_mod = np.diff(bin_edges_L_a_mod)/2 + bin_edges_L_a_mod[0:-1]

os.chdir(f_b)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec_b = np.loadtxt('amplitude_decay.txt')
wav_gro_b = np.loadtxt('wavelength_growth.txt')
t_amp_b = amp_dec_b[0:kb,0]
A_b = amp_dec_b[0:kb,1]
t_wav_b = wav_gro_b[0:kb,0]
L_b = wav_gro_b[0:kb,1]
initial_amp_b, power_amp_b, A_fit_b = power_law_fit(t_amp_b, A_b)
initial_wav_b, power_wav_b, L_fit_b = power_law_fit(t_wav_b, L_b)

L_b_mod, bin_edges_L_b_mod, *_ = stats.binned_statistic(t_wav_b, L_b, 'mean', bins=nnn)
L_b_mod_err, *_ = stats.binned_statistic(t_wav_b, L_b, 'std', bins=nnn)
t_wav_b_mod = np.diff(bin_edges_L_b_mod)/2 + bin_edges_L_b_mod[0:-1]

os.chdir(f_c)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec_c = np.loadtxt('amplitude_decay.txt')
wav_gro_c = np.loadtxt('wavelength_growth.txt')
t_amp_c = amp_dec_c[0:kc,0]
A_c = amp_dec_c[0:kc,1]
t_wav_c = wav_gro_c[0:kc,0]
L_c = wav_gro_c[0:kc,1]
initial_amp_c, power_amp_c, A_fit_c = power_law_fit(t_amp_c, A_c)
initial_wav_c, power_wav_c, L_fit_c = power_law_fit(t_wav_c, L_c)

L_c_mod, bin_edges_L_c_mod, *_ = stats.binned_statistic(t_wav_c, L_c, 'mean', bins=nnn)
L_c_mod_err, *_ = stats.binned_statistic(t_wav_c, L_c, 'std', bins=nnn)
t_wav_c_mod = np.diff(bin_edges_L_c_mod)/2 + bin_edges_L_c_mod[0:-1]

################################################################################

fig = plt.figure()
ax = plt.gca()

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$t/\left( \frac{\eta h_{0}}{\gamma} \right)$', fontsize=15)
plt.ylabel(r'$\frac{L_{\lambda}}{h_{0}}$', fontsize=15)
plt.xlim(100,300000)
plt.xticks([100, 1000, 10000, 100000], [r'$10^{2}$', r'$10^{3}$', r'$10^{4}$', r'$10^{5}$'], fontsize=10)
plt.ylim(9, 100)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.errorbar((t_wav_a_mod)/((h0*mu)/sigma), (L_a_mod*(10**-6))/h0, (L_a_mod_err*(10**-6))/h0, fmt='o', solid_capstyle='projecting', capsize=5, marker=mark, color=col, label=r"%02i $\mu m$, %03i mPa.s, run1" %(round(h0*(10**6)), mu*(10**3)))
plt.errorbar((t_wav_b_mod)/((h0*mu)/sigma), (L_b_mod*(10**-6))/h0, (L_b_mod_err*(10**-6))/h0, fmt='o', solid_capstyle='projecting', capsize=5, marker=mark, color=col, label=r"%02i $\mu m$, %03i mPa.s, run2" %(round(h0*(10**6)), mu*(10**3)))
plt.errorbar((t_wav_c_mod)/((h0*mu)/sigma), (L_c_mod*(10**-6))/h0, (L_c_mod_err*(10**-6))/h0, fmt='o', solid_capstyle='projecting', capsize=5, marker=mark, color=col, label=r"%02i $\mu m$, %03i mPa.s, run3" %(round(h0*(10**6)), mu*(10**3)))

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

plt.show()

################################################################################
