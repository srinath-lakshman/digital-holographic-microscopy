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

# f1 = main_path + r'00050cs0005mum_r1/2018.09.09 16-53'; k = -1
f1 = main_path + r'00050cs0005mum_r2/2018.09.09 16-59'; k = -1
# f1 = main_path + r'00050cs0005mum_r3/2018.09.09 17-07'; k = -1
os.chdir(f1)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec1 = np.loadtxt('amplitude_decay.txt')
wav_gro1 = np.loadtxt('wavelength_growth.txt')
t_amp1 = amp_dec1[0:k,0]
A1 = amp_dec1[0:k,1]
t_wav1 = wav_gro1[0:k,0]
L1 = wav_gro1[0:k,1]
initial_amp1, power_amp1, A_fit1 = power_law_fit(t_amp1, A1)
initial_wav1, power_wav1, L_fit1 = power_law_fit(t_wav1, L1)
# mu1 = int(f1[63:68])
# h01 = int(f1[70:74])
mu1 = 50*(10**-3)
h01 = 5*(10**-6)

L1_mod, bin_edges_L1_mod, *_ = stats.binned_statistic(t_wav1, L1, 'mean', bins=nnn)
L1_mod_err, *_ = stats.binned_statistic(t_wav1, L1, 'std', bins=nnn)
t_wav1_mod = np.diff(bin_edges_L1_mod)/2 + bin_edges_L1_mod[0:-1]

f2 = main_path + r'00050cs0010mum_r1/2018.09.09 17-17'; k = -1
# f2 = main_path + r'00050cs0010mum_r2/2018.09.09 17-23'; k = 650
# f2 = main_path + r'00050cs0010mum_r3/2018.09.09 17-29'; k = -1
os.chdir(f2)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec2 = np.loadtxt('amplitude_decay.txt')
wav_gro2 = np.loadtxt('wavelength_growth.txt')
t_amp2 = amp_dec2[0:k,0]
A2 = amp_dec2[0:k,1]
t_wav2 = wav_gro2[0:k,0]
L2 = wav_gro2[0:k,1]
initial_amp2, power_amp2, A_fit2 = power_law_fit(t_amp2, A2)
initial_wav2, power_wav2, L_fit2 = power_law_fit(t_wav2, L2)
# mu2 = int(f2[63:68])
# h02 = int(f2[70:74])
mu2 = 50*(10**-3)
h02 = 10*(10**-6)

L2_mod, bin_edges_L2_mod, *_ = stats.binned_statistic(t_wav2, L2, 'mean', bins=nnn)
L2_mod_err, *_ = stats.binned_statistic(t_wav2, L2, 'std', bins=nnn)
t_wav2_mod = np.diff(bin_edges_L2_mod)/2 + bin_edges_L2_mod[0:-1]

# f3 = main_path + r'00050cs0015mum_r1/2018.09.09 17-39'; k = 447
f3 = main_path + r'00050cs0015mum_r2/2018.09.09 17-45'; k = 482
# f3 = main_path + r'00050cs0015mum_r3/2018.09.09 17-51'; k = 468
os.chdir(f3)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec3 = np.loadtxt('amplitude_decay.txt')
wav_gro3 = np.loadtxt('wavelength_growth.txt')
t_amp3 = amp_dec3[0:k,0]
A3 = amp_dec3[0:k,1]
t_wav3 = wav_gro3[0:k,0]
L3 = wav_gro3[0:k,1]
initial_amp3, power_amp3, A_fit3 = power_law_fit(t_amp3, A3)
initial_wav3, power_wav3, L_fit3 = power_law_fit(t_wav3, L3)
# mu3 = int(f3[63:68])
# h03 = int(f3[70:74])
mu3 = 50*(10**-3)
h03 = 15*(10**-6)

L3_mod, bin_edges_L3_mod, *_ = stats.binned_statistic(t_wav3, L3, 'mean', bins=nnn)
L3_mod_err, *_ = stats.binned_statistic(t_wav3, L3, 'std', bins=nnn)
t_wav3_mod = np.diff(bin_edges_L3_mod)/2 + bin_edges_L3_mod[0:-1]

f4 = main_path + r'00100cs0005mum_r1/2018.09.09 18-39'; k = -1
# f4 = main_path + r'00100cs0005mum_r2/2018.09.09 18-43'; k = -1
# f4 = main_path + r'00100cs0005mum_r3/2018.09.09 18-45'; k = -1
os.chdir(f4)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec4 = np.loadtxt('amplitude_decay.txt')
wav_gro4 = np.loadtxt('wavelength_growth.txt')
t_amp4 = amp_dec4[0:k,0]
A4 = amp_dec4[0:k,1]
t_wav4 = wav_gro4[0:k,0]
L4 = wav_gro4[0:k,1]
initial_amp4, power_amp4, A_fit4 = power_law_fit(t_amp4, A4)
initial_wav4, power_wav4, L_fit4 = power_law_fit(t_wav4, L4)
# mu4 = int(f4[63:68])
# h04 = int(f4[70:74])
mu4 = 100*(10**-3)
h04 = 5*(10**-6)

L4_mod, bin_edges_L4_mod, *_ = stats.binned_statistic(t_wav4, L4, 'mean', bins=nnn)
L4_mod_err, *_ = stats.binned_statistic(t_wav4, L4, 'std', bins=nnn)
t_wav4_mod = np.diff(bin_edges_L4_mod)/2 + bin_edges_L4_mod[0:-1]

f5 = main_path + r'00100cs0010mum_r1/2018.09.09 18-53'; k = -1
os.chdir(f5)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec5 = np.loadtxt('amplitude_decay.txt')
wav_gro5 = np.loadtxt('wavelength_growth.txt')
t_amp5 = amp_dec5[0:k,0]
A5 = amp_dec5[0:k,1]
t_wav5 = wav_gro5[0:k,0]
L5 = wav_gro5[0:k,1]
initial_amp5, power_amp5, A_fit5 = power_law_fit(t_amp5, A5)
initial_wav5, power_wav5, L_fit5 = power_law_fit(t_wav5, L5)
# mu5 = int(f5[63:68])
# h05 = int(f5[70:74])
mu5 = 100*(10**-3)
h05 = 10*(10**-6)

L5_mod, bin_edges_L5_mod, *_ = stats.binned_statistic(t_wav5, L5, 'mean', bins=nnn)
L5_mod_err, *_ = stats.binned_statistic(t_wav5, L5, 'std', bins=nnn)
t_wav5_mod = np.diff(bin_edges_L5_mod)/2 + bin_edges_L5_mod[0:-1]

f6 = main_path + r'00100cs0015mum_r3/2018.09.09 19-24'; k = 733
os.chdir(f6)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec6 = np.loadtxt('amplitude_decay.txt')
wav_gro6 = np.loadtxt('wavelength_growth.txt')
t_amp6 = amp_dec6[0:k,0]
A6 = amp_dec6[0:k,1]
t_wav6 = wav_gro6[0:k,0]
L6 = wav_gro6[0:k,1]
initial_amp6, power_amp6, A_fit6 = power_law_fit(t_amp6, A6)
initial_wav6, power_wav6, L_fit6 = power_law_fit(t_wav6, L6)
# mu6 = int(f6[63:68])
# h06 = int(f6[70:74])
mu6 = 100*(10**-3)
h06 = 15*(10**-6)

L6_mod, bin_edges_L6_mod, *_ = stats.binned_statistic(t_wav6, L6, 'mean', bins=nnn)
L6_mod_err, *_ = stats.binned_statistic(t_wav6, L6, 'std', bins=nnn)
t_wav6_mod = np.diff(bin_edges_L6_mod)/2 + bin_edges_L6_mod[0:-1]

# f7 = main_path + r'00200cs0005mum_r1/2018.09.09 20-17'; k = -1
f7 = main_path + r'00200cs0005mum_r2/2018.09.09 20-20'; k = -1
# f7 = main_path + r'00200cs0005mum_r3/2018.09.09 20-27'; k = -1
os.chdir(f7)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec7 = np.loadtxt('amplitude_decay.txt')
wav_gro7 = np.loadtxt('wavelength_growth.txt')
t_amp7 = amp_dec7[0:k,0]
A7 = amp_dec7[0:k,1]
t_wav7 = wav_gro7[0:k,0]
L7 = wav_gro7[0:k,1]
initial_amp7, power_amp7, A_fit7 = power_law_fit(t_amp7, A7)
initial_wav7, power_wav7, L_fit7 = power_law_fit(t_wav7, L7)
# mu7 = int(f7[63:68])
# h07 = int(f7[70:74])
mu7 = 200*(10**-3)
h07 = 5*(10**-6)

L7_mod, bin_edges_L7_mod, *_ = stats.binned_statistic(t_wav7, L7, 'mean', bins=nnn)
L7_mod_err, *_ = stats.binned_statistic(t_wav7, L7, 'std', bins=nnn)
t_wav7_mod = np.diff(bin_edges_L7_mod)/2 + bin_edges_L7_mod[0:-1]

f8 = main_path + r'00200cs0010mum_r1/2018.09.09 20-34'; k = 1416
f8 = main_path + r'00200cs0010mum_r3_TODO/2018.09.09 20-40'; k = 1416
os.chdir(f8)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec8 = np.loadtxt('amplitude_decay.txt')
wav_gro8 = np.loadtxt('wavelength_growth.txt')
t_amp8 = amp_dec8[0:k,0]
A8 = amp_dec8[0:k,1]
t_wav8 = wav_gro8[0:k,0]
L8 = wav_gro8[0:k,1]
initial_amp8, power_amp8, A_fit8 = power_law_fit(t_amp8, A8)
initial_wav8, power_wav8, L_fit8 = power_law_fit(t_wav8, L8)
# mu8 = int(f8[63:68])
# h08 = int(f8[70:74])
mu8 = 200*(10**-3)
h08 = 10*(10**-6)

L8_mod, bin_edges_L8_mod, *_ = stats.binned_statistic(t_wav8, L8, 'mean', bins=nnn)
L8_mod_err, *_ = stats.binned_statistic(t_wav8, L8, 'std', bins=nnn)
t_wav8_mod = np.diff(bin_edges_L8_mod)/2 + bin_edges_L8_mod[0:-1]

f9 = main_path + r'00200cs0015mum_r1/2018.09.09 20-47'; k = -1
os.chdir(f9)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec9 = np.loadtxt('amplitude_decay.txt')
wav_gro9 = np.loadtxt('wavelength_growth.txt')
t_amp9 = amp_dec9[0:k,0]
A9 = amp_dec9[0:k,1]
t_wav9 = wav_gro9[0:k,0]
L9 = wav_gro9[0:k,1]
initial_amp9, power_amp9, A_fit9 = power_law_fit(t_amp9, A9)
initial_wav9, power_wav9, L_fit9 = power_law_fit(t_wav9, L9)
# mu9 = int(f9[63:68])
# h09 = int(f9[70:74])
mu9 = 200*(10**-3)
h09 = 15*(10**-6)

L9_mod, bin_edges_L9_mod, *_ = stats.binned_statistic(t_wav9, L9, 'mean', bins=nnn)
L9_mod_err, *_ = stats.binned_statistic(t_wav9, L9, 'std', bins=nnn)
t_wav9_mod = np.diff(bin_edges_L9_mod)/2 + bin_edges_L9_mod[0:-1]

################################################################################

x_coor = np.concatenate(((t_wav1_mod)/((h01*mu1)/sigma), (t_wav2_mod)/((h02*mu2)/sigma), (t_wav3_mod)/((h03*mu3)/sigma), \
                         (t_wav4_mod)/((h04*mu4)/sigma), (t_wav5_mod)/((h05*mu5)/sigma), (t_wav6_mod)/((h06*mu6)/sigma), \
                         (t_wav7_mod)/((h07*mu7)/sigma), (t_wav8_mod)/((h08*mu8)/sigma), (t_wav9_mod)/((h09*mu9)/sigma)))
y_coor = np.concatenate(((L1_mod*(10**-6))/h01, (L2_mod*(10**-6))/h02, (L3_mod*(10**-6))/h03, \
                         (L4_mod*(10**-6))/h04, (L5_mod*(10**-6))/h05, (L6_mod*(10**-6))/h06, \
                         (L7_mod*(10**-6))/h07, (L8_mod*(10**-6))/h08, (L9_mod*(10**-6))/h09))

initial_coor, power_coor, y_fit_coor = power_law_fit(x_coor,y_coor)

print(initial_coor, power_coor)
input()

################################################################################

fig = plt.figure()
ax = plt.gca()

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$t/\left( \eta h_{0} / \gamma \right)$', fontsize=15)
plt.ylabel(r'$L_{\lambda} (t) / h_{0}$', fontsize=15)
plt.xlim(100,300000)
plt.xticks([100, 1000, 10000, 100000], [r'$10^{2}$', r'$10^{3}$', r'$10^{4}$', r'$10^{5}$'], fontsize=15)
plt.ylim(9, 100)
plt.yticks([10, 100], fontsize=15)
# yticks = ax.yaxis.get_major_ticks()
# for i in range(0,len(yticks)):
#     yticks[i].label1.set_visible(False)
#
# ax.yaxis.set_minor_locator(plt.FixedLocator([9,80]))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax2 = plt.add_axes([0.25*box.x0, 0.6*box.y0, 0.2*box.width * 0.8, 0.2*box.height])

# plt.plot([1, (100/initial_coor)**(1/power_coor)],[initial_coor, 100],'--',color='black')
plt.plot([100, 1000000],[10, 100],'--',color='black')

# plt.ylim(5, 100)
# plt.yticks([5, 10, 100], fontsize=12.5)
plt.errorbar((t_wav1_mod-t_wav1_mod[0])/((h01*mu1)/sigma), (L1_mod*(10**-6))/h01, (L1_mod_err*(10**-6))/h01, fmt='o', solid_capstyle='projecting', capsize=5, marker="v", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h01*(10**6)), mu1*(10**3)))
plt.errorbar((t_wav2_mod-t_wav2_mod[0])/((h02*mu2)/sigma), (L2_mod*(10**-6))/h02, (L2_mod_err*(10**-6))/h02, fmt='o', solid_capstyle='projecting', capsize=5, marker="o", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h02*(10**6)), mu2*(10**3)))
plt.errorbar((t_wav3_mod-t_wav3_mod[0])/((h03*mu3)/sigma), (L3_mod*(10**-6))/h03, (L3_mod_err*(10**-6))/h03, fmt='o', solid_capstyle='projecting', capsize=5, marker="^", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h03*(10**6)), mu3*(10**3)))
plt.errorbar((t_wav4_mod-t_wav4_mod[0])/((h04*mu4)/sigma), (L4_mod*(10**-6))/h04, (L4_mod_err*(10**-6))/h04, fmt='o', solid_capstyle='projecting', capsize=5, marker="v", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h04*(10**6)), mu4*(10**3)))
plt.errorbar((t_wav5_mod-t_wav5_mod[0])/((h05*mu5)/sigma), (L5_mod*(10**-6))/h05, (L5_mod_err*(10**-6))/h05, fmt='o', solid_capstyle='projecting', capsize=5, marker="o", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h05*(10**6)), mu5*(10**3)))
plt.errorbar((t_wav6_mod-t_wav6_mod[0])/((h06*mu6)/sigma), (L6_mod*(10**-6))/h06, (L6_mod_err*(10**-6))/h06, fmt='o', solid_capstyle='projecting', capsize=5, marker="^", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h06*(10**6)), mu6*(10**3)))
plt.errorbar((t_wav7_mod-t_wav7_mod[0])/((h07*mu7)/sigma), (L7_mod*(10**-6))/h07, (L7_mod_err*(10**-6))/h07, fmt='o', solid_capstyle='projecting', capsize=5, marker="v", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h07*(10**6)), mu7*(10**3)))
plt.errorbar((t_wav8_mod-t_wav8_mod[0])/((h08*mu8)/sigma), (L8_mod*(10**-6))/h08, (L8_mod_err*(10**-6))/h08, fmt='o', solid_capstyle='projecting', capsize=5, marker="o", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h08*(10**6)), mu8*(10**3)))
plt.errorbar((t_wav9_mod-t_wav9_mod[0])/((h09*mu9)/sigma), (L9_mod*(10**-6))/h09, (L9_mod_err*(10**-6))/h09, fmt='o', solid_capstyle='projecting', capsize=5, marker="^", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h09*(10**6)), mu9*(10**3)))

plt.plot([1000,100000,100000,1000],[12,12*np.sqrt(10),12,12], color="black")
plt.text(10000,10,'4',fontsize='15')
plt.text(125000,20,'1',fontsize='15')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

axicon = fig.add_axes([0.2,0.525,0.2,0.3])
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.01,4)
# plt.xticks([0.01, 0.1, 1, 4], [r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'4 $\times$ $10^{0}$'])
plt.xticks([0.01, 0.1, 1, 4], [0.01, 0.1, 1, 4])
plt.ylim(60,700)
# plt.yticks([60, 100, 700],[r'6 $\times$ $10^{1}$', r'$10^{2}$', r'7 $\times$ $10^{2}$'])
plt.yticks([60, 100, 700],[60, 100, 700])
plt.xlabel(r'$t$ [s]', fontsize=10)
plt.ylabel(r'$L_{\lambda}$ [$\mu m$]', fontsize=10)
plt.errorbar(t_wav1_mod, L1_mod, L1_mod_err, fmt='o', solid_capstyle='projecting', capsize=5, marker="v", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h01*(10**6)), mu1*(10**3)))
plt.errorbar(t_wav2_mod, L2_mod, L2_mod_err, fmt='o', solid_capstyle='projecting', capsize=5, marker="o", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h02*(10**6)), mu2*(10**3)))
plt.errorbar(t_wav3_mod, L3_mod, L3_mod_err, fmt='o', solid_capstyle='projecting', capsize=5, marker="^", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h03*(10**6)), mu3*(10**3)))
plt.errorbar(t_wav4_mod, L4_mod, L4_mod_err, fmt='o', solid_capstyle='projecting', capsize=5, marker="v", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h04*(10**6)), mu4*(10**3)))
plt.errorbar(t_wav5_mod, L5_mod, L5_mod_err, fmt='o', solid_capstyle='projecting', capsize=5, marker="o", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h05*(10**6)), mu5*(10**3)))
plt.errorbar(t_wav6_mod, L6_mod, L6_mod_err, fmt='o', solid_capstyle='projecting', capsize=5, marker="^", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h06*(10**6)), mu6*(10**3)))
plt.errorbar(t_wav7_mod, L7_mod, L7_mod_err, fmt='o', solid_capstyle='projecting', capsize=5, marker="v", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h07*(10**6)), mu7*(10**3)))
plt.errorbar(t_wav8_mod, L8_mod, L8_mod_err, fmt='o', solid_capstyle='projecting', capsize=5, marker="o", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h08*(10**6)), mu8*(10**3)))
plt.errorbar(t_wav9_mod, L9_mod, L9_mod_err, fmt='o', solid_capstyle='projecting', capsize=5, marker="^", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h09*(10**6)), mu9*(10**3)))

plt.show()

################################################################################

os.chdir('/home/devici/Desktop/wavelength_scaling')

np.savetxt('00050cs0005mum_scaled_time.txt',(t_wav1_mod)/((h01*mu1)/sigma))
np.savetxt('00050cs0010mum_scaled_time.txt',(t_wav2_mod)/((h02*mu2)/sigma))
np.savetxt('00050cs0015mum_scaled_time.txt',(t_wav3_mod)/((h03*mu3)/sigma))
np.savetxt('00100cs0005mum_scaled_time.txt',(t_wav4_mod)/((h04*mu4)/sigma))
np.savetxt('00100cs0010mum_scaled_time.txt',(t_wav5_mod)/((h05*mu5)/sigma))
np.savetxt('00100cs0015mum_scaled_time.txt',(t_wav6_mod)/((h06*mu6)/sigma))
np.savetxt('00200cs0005mum_scaled_time.txt',(t_wav7_mod)/((h07*mu7)/sigma))
np.savetxt('00200cs0010mum_scaled_time.txt',(t_wav8_mod)/((h08*mu8)/sigma))
np.savetxt('00200cs0015mum_scaled_time.txt',(t_wav9_mod)/((h09*mu9)/sigma))

np.savetxt('00050cs0005mum_scaled_wav.txt',(L1_mod*(10**-6))/h01)
np.savetxt('00050cs0010mum_scaled_wav.txt',(L2_mod*(10**-6))/h02)
np.savetxt('00050cs0015mum_scaled_wav.txt',(L3_mod*(10**-6))/h03)
np.savetxt('00100cs0005mum_scaled_wav.txt',(L4_mod*(10**-6))/h04)
np.savetxt('00100cs0010mum_scaled_wav.txt',(L5_mod*(10**-6))/h05)
np.savetxt('00100cs0015mum_scaled_wav.txt',(L6_mod*(10**-6))/h06)
np.savetxt('00200cs0005mum_scaled_wav.txt',(L7_mod*(10**-6))/h07)
np.savetxt('00200cs0010mum_scaled_wav.txt',(L8_mod*(10**-6))/h08)
np.savetxt('00200cs0015mum_scaled_wav.txt',(L9_mod*(10**-6))/h09)

np.savetxt('00050cs0005mum_scaled_errorbar.txt',(L1_mod_err*(10**-6))/h01)
np.savetxt('00050cs0010mum_scaled_errorbar.txt',(L2_mod_err*(10**-6))/h02)
np.savetxt('00050cs0015mum_scaled_errorbar.txt',(L3_mod_err*(10**-6))/h03)
np.savetxt('00100cs0005mum_scaled_errorbar.txt',(L4_mod_err*(10**-6))/h04)
np.savetxt('00100cs0010mum_scaled_errorbar.txt',(L5_mod_err*(10**-6))/h05)
np.savetxt('00100cs0015mum_scaled_errorbar.txt',(L6_mod_err*(10**-6))/h06)
np.savetxt('00200cs0005mum_scaled_errorbar.txt',(L7_mod_err*(10**-6))/h07)
np.savetxt('00200cs0010mum_scaled_errorbar.txt',(L8_mod_err*(10**-6))/h08)
np.savetxt('00200cs0015mum_scaled_errorbar.txt',(L9_mod_err*(10**-6))/h09)

np.savetxt('00050cs0005mum_time.txt',t_wav1_mod)
np.savetxt('00050cs0010mum_time.txt',t_wav2_mod)
np.savetxt('00050cs0015mum_time.txt',t_wav3_mod)
np.savetxt('00100cs0005mum_time.txt',t_wav4_mod)
np.savetxt('00100cs0010mum_time.txt',t_wav5_mod)
np.savetxt('00100cs0015mum_time.txt',t_wav6_mod)
np.savetxt('00200cs0005mum_time.txt',t_wav7_mod)
np.savetxt('00200cs0010mum_time.txt',t_wav8_mod)
np.savetxt('00200cs0015mum_time.txt',t_wav9_mod)

np.savetxt('00050cs0005mum_wav.txt',L1_mod)
np.savetxt('00050cs0010mum_wav.txt',L2_mod)
np.savetxt('00050cs0015mum_wav.txt',L3_mod)
np.savetxt('00100cs0005mum_wav.txt',L4_mod)
np.savetxt('00100cs0010mum_wav.txt',L5_mod)
np.savetxt('00100cs0015mum_wav.txt',L6_mod)
np.savetxt('00200cs0005mum_wav.txt',L7_mod)
np.savetxt('00200cs0010mum_wav.txt',L8_mod)
np.savetxt('00200cs0015mum_wav.txt',L9_mod)

np.savetxt('00050cs0005mum_errorbar.txt',L1_mod_err)
np.savetxt('00050cs0010mum_errorbar.txt',L2_mod_err)
np.savetxt('00050cs0015mum_errorbar.txt',L3_mod_err)
np.savetxt('00100cs0005mum_errorbar.txt',L4_mod_err)
np.savetxt('00100cs0010mum_errorbar.txt',L5_mod_err)
np.savetxt('00100cs0015mum_errorbar.txt',L6_mod_err)
np.savetxt('00200cs0005mum_errorbar.txt',L7_mod_err)
np.savetxt('00200cs0010mum_errorbar.txt',L8_mod_err)
np.savetxt('00200cs0015mum_errorbar.txt',L9_mod_err)
