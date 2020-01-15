import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
from FUNC_ import power_law_fit
from FUNC_ import sphere_to_pancake
from FUNC_ import binning_data
from scipy import stats
from scipy import integrate

################################################################################

conv = 2.967841e-06
sigma = 20.3/1000
nnn = 100
pp = 2

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

A1_mod, bin_edges_A1_mod, *_ = stats.binned_statistic(t_amp1, A1, 'mean', bins=nnn)
A1_mod_err, *_ = stats.binned_statistic(t_amp1, A1, 'std', bins=nnn)
t_amp1_mod = np.diff(bin_edges_A1_mod)/2 + bin_edges_A1_mod[0:-1]

avg_A1_mod = np.mean(A1_mod[-pp:-1])
avg_t_amp1_mod = np.mean(t_amp1_mod[-pp:-1])
tc_1 = ((avg_A1_mod/A1_mod[0])**2)*(avg_t_amp1_mod)

os.chdir('..')
os.chdir('..')
os.chdir(os.getcwd() + r'/Unwrapped')

def_prof1 = np.loadtxt('h_00674.txt')
def_prof1 = np.loadtxt('h_00679.txt')
r1 = np.loadtxt('r.txt')

moment_def_1 = np.trapz((r1-(0.68568/1000))*def_prof1,dx=conv)*(conv**1)*(10**9)

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

A2_mod, bin_edges_A2_mod, *_ = stats.binned_statistic(t_amp2, A2, 'mean', bins=nnn)
A2_mod_err, *_ = stats.binned_statistic(t_amp2, A2, 'std', bins=nnn)
t_amp2_mod = np.diff(bin_edges_A2_mod)/2 + bin_edges_A2_mod[0:-1]

avg_A2_mod = np.mean(A2_mod[-pp:-1])
avg_t_amp2_mod = np.mean(t_amp2_mod[-pp:-1])
tc_2 = ((avg_A2_mod/A2_mod[0])**2)*(avg_t_amp2_mod)

os.chdir('..')
os.chdir('..')
os.chdir(os.getcwd() + r'/Unwrapped')

def_prof2 = np.loadtxt('h_00444.txt')
def_prof2 = np.loadtxt('h_00449.txt')
r2 = np.loadtxt('r.txt')

moment_def_2 = np.trapz((r2-(0.68568/1000))*def_prof2,dx=conv)*(conv**1)*(10**9)

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

A3_mod, bin_edges_A3_mod, *_ = stats.binned_statistic(t_amp3, A3, 'mean', bins=nnn)
A3_mod_err, *_ = stats.binned_statistic(t_amp3, A3, 'std', bins=nnn)
t_amp3_mod = np.diff(bin_edges_A3_mod)/2 + bin_edges_A3_mod[0:-1]

avg_A3_mod = np.mean(A3_mod[-pp:-1])
avg_t_amp3_mod = np.mean(t_amp3_mod[-pp:-1])
tc_3 = ((avg_A3_mod/A3_mod[0])**2)*(avg_t_amp3_mod)

os.chdir('..')
os.chdir('..')
os.chdir(os.getcwd() + r'/Unwrapped')

def_prof3 = np.loadtxt('h_00357.txt')
def_prof3 = np.loadtxt('h_00362.txt')
r3 = np.loadtxt('r.txt')

moment_def_3 = np.trapz((r3-(0.68568/1000))*def_prof3,dx=conv)*(conv**1)*(10**9)

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

A4_mod, bin_edges_A4_mod, *_ = stats.binned_statistic(t_amp4, A4, 'mean', bins=nnn)
A4_mod_err, *_ = stats.binned_statistic(t_amp4, A4, 'std', bins=nnn)
t_amp4_mod = np.diff(bin_edges_A4_mod)/2 + bin_edges_A4_mod[0:-1]

avg_A4_mod = np.mean(A4_mod[-pp:-1])
avg_t_amp4_mod = np.mean(t_amp4_mod[-pp:-1])
tc_4 = ((avg_A4_mod/A4_mod[0])**2)*(avg_t_amp4_mod)

os.chdir('..')
os.chdir('..')
os.chdir(os.getcwd() + r'/Unwrapped')

def_prof4 = np.loadtxt('h_01020.txt')
def_prof4 = np.loadtxt('h_01025.txt')
r4 = np.loadtxt('r.txt')

moment_def_4 = np.trapz((r4-(0.68568/1000))*def_prof4,dx=conv)*(conv**1)*(10**9)

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

A5_mod, bin_edges_A5_mod, *_ = stats.binned_statistic(t_amp5, A5, 'mean', bins=nnn)
A5_mod_err, *_ = stats.binned_statistic(t_amp5, A5, 'std', bins=nnn)
t_amp5_mod = np.diff(bin_edges_A5_mod)/2 + bin_edges_A5_mod[0:-1]

avg_A5_mod = np.mean(A5_mod[-pp:-1])
avg_t_amp5_mod = np.mean(t_amp5_mod[-pp:-1])
tc_5 = ((avg_A5_mod/A5_mod[0])**2)*(avg_t_amp5_mod)

os.chdir('..')
os.chdir('..')
os.chdir(os.getcwd() + r'/Unwrapped')

def_prof5 = np.loadtxt('h_00964.txt')
def_prof5 = np.loadtxt('h_00969.txt')
r5 = np.loadtxt('r.txt')

moment_def_5 = np.trapz((r5-(0.68568/1000))*def_prof5,dx=conv)*(conv**1)*(10**9)

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

A6_mod, bin_edges_A6_mod, *_ = stats.binned_statistic(t_amp6, A6, 'mean', bins=nnn)
A6_mod_err, *_ = stats.binned_statistic(t_amp6, A6, 'std', bins=nnn)
t_amp6_mod = np.diff(bin_edges_A6_mod)/2 + bin_edges_A6_mod[0:-1]

avg_A6_mod = np.mean(A6_mod[-pp:-1])
avg_t_amp6_mod = np.mean(t_amp6_mod[-pp:-1])
tc_6 = ((avg_A6_mod/A6_mod[0])**2)*(avg_t_amp6_mod)

os.chdir('..')
os.chdir('..')
os.chdir(os.getcwd() + r'/Unwrapped')

def_prof6 = np.loadtxt('h_00430.txt')
def_prof6 = np.loadtxt('h_00435.txt')
r6 = np.loadtxt('r.txt')

moment_def_6 = np.trapz((r6-(0.68568/1000))*def_prof6,dx=conv)*(conv**1)*(10**9)

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

A7_mod, bin_edges_A7_mod, *_ = stats.binned_statistic(t_amp7, A7, 'mean', bins=nnn)
A7_mod_err, *_ = stats.binned_statistic(t_amp7, A7, 'std', bins=nnn)
t_amp7_mod = np.diff(bin_edges_A7_mod)/2 + bin_edges_A7_mod[0:-1]

avg_A7_mod = np.mean(A7_mod[-pp:-1])
avg_t_amp7_mod = np.mean(t_amp7_mod[-pp:-1])
tc_7 = ((avg_A7_mod/A7_mod[0])**2)*(avg_t_amp7_mod)

os.chdir('..')
os.chdir('..')
os.chdir(os.getcwd() + r'/Unwrapped')

def_prof7 = np.loadtxt('h_00263.txt')
def_prof7 = np.loadtxt('h_00268.txt')
r7 = np.loadtxt('r.txt')

moment_def_7 = np.trapz((r7-(0.68568/1000))*def_prof7,dx=conv)*(conv**1)*(10**9)

f8 = main_path + r'00200cs0010mum_r1/2018.09.09 20-34'; k = 1416
# f8 = main_path + r'00200cs0010mum_r3_TODO/2018.09.09 20-40'; k = 1416
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

A8_mod, bin_edges_A8_mod, *_ = stats.binned_statistic(t_amp8, A8, 'mean', bins=nnn)
A8_mod_err, *_ = stats.binned_statistic(t_amp8, A8, 'std', bins=nnn)
t_amp8_mod = np.diff(bin_edges_A8_mod)/2 + bin_edges_A8_mod[0:-1]

avg_A8_mod = np.mean(A8_mod[-pp:-1])
avg_t_amp8_mod = np.mean(t_amp8_mod[-pp:-1])
tc_8 = ((avg_A8_mod/A8_mod[0])**2)*(avg_t_amp8_mod)

os.chdir('..')
os.chdir('..')
os.chdir(os.getcwd() + r'/Unwrapped')

def_prof8 = np.loadtxt('h_00568.txt')
def_prof8 = np.loadtxt('h_00573.txt')
r8 = np.loadtxt('r.txt')

moment_def_8 = np.trapz((r8-(0.68568/1000))*def_prof8,dx=conv)*(conv**1)*(10**9)

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

A9_mod, bin_edges_A9_mod, *_ = stats.binned_statistic(t_amp9, A9, 'mean', bins=nnn)
A9_mod_err, *_ = stats.binned_statistic(t_amp9, A9, 'std', bins=nnn)
t_amp9_mod = np.diff(bin_edges_A9_mod)/2 + bin_edges_A9_mod[0:-1]

avg_A9_mod = np.mean(A9_mod[-pp:-1])
avg_t_amp9_mod = np.mean(t_amp9_mod[-pp:-1])
tc_9 = ((avg_A9_mod/A9_mod[0])**2)*(avg_t_amp9_mod)

os.chdir('..')
os.chdir('..')
os.chdir(os.getcwd() + r'/Unwrapped')

def_prof9 = np.loadtxt('h_00486.txt')
def_prof9 = np.loadtxt('h_00491.txt')
r9 = np.loadtxt('r.txt')

moment_def_9 = np.trapz((r9-(0.68568/1000))*def_prof9,dx=conv)*(conv**1)*(10**9)

################################################################################

fig = plt.figure(1)
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')

plt.xlim(0.04,80)
plt.ylim(0.1,1.2)
plt.xticks([0.04, 0.1, 1, 10, 80], fontsize=15)
plt.yticks([0.1, 1], fontsize=15)

plt.xlabel(r'$\left( t - t_{0} \right)/t_{c}$', fontsize=15)
plt.ylabel(r'$A_{\delta} (t) / A_{\delta} (0)$', fontsize=15)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
# plt.xlabel(r'$\left( t - t_{0} \right)/\left( \frac{\eta h_{0}}{\gamma} \right)$', fontsize=25)
# plt.ylabel(r'$\frac{A_{\delta}}{A_{\delta,0}}$', fontsize=25)
# plt.scatter((t_amp1-t_amp1[0])/((h01*mu1)/sigma), (A1*(10**-9))/((A1[0])*(10**-9)), marker=".", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h01*(10**6)), mu1*(10**3)))
# plt.scatter((t_amp2-t_amp2[0])/((h02*mu2)/sigma), (A2*(10**-9))/(max(A2)*(10**-9)), marker="+", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h02*(10**6)), mu2*(10**3)))
# plt.scatter((t_amp3-t_amp3[0])/((h03*mu3)/sigma), (A3*(10**-9))/(max(A3)*(10**-9)), marker="x", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h03*(10**6)), mu3*(10**3)))
# plt.scatter((t_amp4-t_amp4[0])/((h04*mu4)/sigma), (A4*(10**-9))/((A4[0])*(10**-9)), marker=".", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h04*(10**6)), mu4*(10**3)))
# plt.scatter((t_amp5-t_amp5[0])/((h05*mu5)/sigma), (A5*(10**-9))/(max(A5)*(10**-9)), marker="+", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h05*(10**6)), mu5*(10**3)))
# plt.scatter((t_amp6-t_amp6[0])/((h06*mu6)/sigma), (A6*(10**-9))/(max(A6)*(10**-9)), marker="x", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h06*(10**6)), mu6*(10**3)))
# plt.scatter((t_amp7-t_amp7[0])/((h07*mu7)/sigma), (A7*(10**-9))/((A7[0])*(10**-9)), marker=".", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h07*(10**6)), mu7*(10**3)))
# plt.scatter((t_amp8-t_amp8[0])/((h08*mu8)/sigma), (A8*(10**-9))/(max(A8)*(10**-9)), marker="+", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h08*(10**6)), mu8*(10**3)))
# plt.scatter((t_amp9-t_amp7[0])/((h09*mu9)/sigma), (A9*(10**-9))/(max(A9)*(10**-9)), marker="x", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h09*(10**6)), mu9*(10**3)))

# plt.scatter((t_amp1-t_amp1[0])/((h01*mu1)/sigma), (A1_mod*(10**-9))/(max(A1)*(10**-9)), marker=".", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h01*(10**6)), mu1*(10**3)))

# plt.scatter(t_amp1_mod, A1_mod)

# h01=10*(10**-6); h02=10*(10**-6); h03=10*(10**-6); h04=10*(10**-6); h05=10*(10**-6); h06=10*(10**-6); h07=10*(10**-6); h08=10*(10**-6); h09=10*(10**-6);
# mu1=100*(10**-3); mu2=100*(10**-3); mu3=100*(10**-3); mu4=100*(10**-3); mu5=100*(10**-3); mu6=100*(10**-3); mu7=100*(10**-3); mu8=100*(10**-3); mu9=100*(10**-3);

# print(t_amp1_mod[0])
# print(t_amp4_mod[0])
# print(t_amp7_mod[0])
#
# input('')

set_pt = 0
pp = 1

plt.errorbar((t_amp1_mod[set_pt+1:-1] - t_amp1_mod[set_pt])/tc_1, (A1_mod[set_pt+1:-1]*(10**-9))/((A1_mod[set_pt])*(10**-9)), (A1_mod_err[set_pt+1:-1]*(10**-9))/((A1_mod[set_pt])*(10**-9)), fmt='o', solid_capstyle='projecting', capsize=5, marker="v", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h01*(10**6)), mu1*(10**3)))
plt.errorbar((t_amp2_mod[set_pt+1:-1] - t_amp2_mod[set_pt])/tc_2, (A2_mod[set_pt+1:-1]*(10**-9))/((A2_mod[set_pt])*(10**-9)), (A2_mod_err[set_pt+1:-1]*(10**-9))/((A2_mod[set_pt])*(10**-9)), fmt='o', solid_capstyle='projecting', capsize=5, marker="o", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h02*(10**6)), mu2*(10**3)))
plt.errorbar((t_amp3_mod[set_pt+1:-1] - t_amp3_mod[set_pt])/tc_3, (A3_mod[set_pt+1:-1]*(10**-9))/((A3_mod[set_pt])*(10**-9)), (A3_mod_err[set_pt+1:-1]*(10**-9))/((A3_mod[set_pt])*(10**-9)), fmt='o', solid_capstyle='projecting', capsize=5, marker="^", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h03*(10**6)), mu3*(10**3)))
plt.errorbar((t_amp4_mod[set_pt+1:-1] - t_amp4_mod[set_pt])/tc_4, (A4_mod[set_pt+1:-1]*(10**-9))/((A4_mod[set_pt])*(10**-9)), (A4_mod_err[set_pt+1:-1]*(10**-9))/((A4_mod[set_pt])*(10**-9)), fmt='o', solid_capstyle='projecting', capsize=5, marker="v", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h04*(10**6)), mu4*(10**3)))
plt.errorbar((t_amp5_mod[set_pt+1:-1] - t_amp5_mod[set_pt])/tc_5, (A5_mod[set_pt+1:-1]*(10**-9))/((A5_mod[set_pt])*(10**-9)), (A5_mod_err[set_pt+1:-1]*(10**-9))/((A5_mod[set_pt])*(10**-9)), fmt='o', solid_capstyle='projecting', capsize=5, marker="o", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h05*(10**6)), mu5*(10**3)))
plt.errorbar((t_amp6_mod[set_pt+1:-1] - t_amp6_mod[set_pt])/tc_6, (A6_mod[set_pt+1:-1]*(10**-9))/((A6_mod[set_pt])*(10**-9)), (A6_mod_err[set_pt+1:-1]*(10**-9))/((A6_mod[set_pt])*(10**-9)), fmt='o', solid_capstyle='projecting', capsize=5, marker="^", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h06*(10**6)), mu6*(10**3)))
plt.errorbar((t_amp7_mod[set_pt+1:-1] - t_amp7_mod[set_pt])/tc_7, (A7_mod[set_pt+1:-1]*(10**-9))/((A7_mod[set_pt])*(10**-9)), (A7_mod_err[set_pt+1:-1]*(10**-9))/((A7_mod[set_pt])*(10**-9)), fmt='o', solid_capstyle='projecting', capsize=5, marker="v", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h07*(10**6)), mu7*(10**3)))
plt.errorbar((t_amp8_mod[set_pt+1:-1] - t_amp8_mod[set_pt])/tc_8, (A8_mod[set_pt+1:-1]*(10**-9))/((A8_mod[set_pt])*(10**-9)), (A8_mod_err[set_pt+1:-1]*(10**-9))/((A8_mod[set_pt])*(10**-9)), fmt='o', solid_capstyle='projecting', capsize=5, marker="o", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h08*(10**6)), mu8*(10**3)))
plt.errorbar((t_amp9_mod[set_pt+1:-1] - t_amp9_mod[set_pt])/tc_9, (A9_mod[set_pt+1:-1]*(10**-9))/((A9_mod[set_pt])*(10**-9)), (A9_mod_err[set_pt+1:-1]*(10**-9))/((A9_mod[set_pt])*(10**-9)), fmt='o', solid_capstyle='projecting', capsize=5, marker="^", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h09*(10**6)), mu9*(10**3)))

plt.axhline(1, color='k',linestyle='--')
plt.plot([0.1, 100],[(0.1)**(-0.5), 0.1], color='k', linestyle='--')
# plt.text(2, 0.8, r'$ \sim \left( t / t_{c} \right)^{-1/2}$', fontsize=16)
plt.plot([40,40,40*(6.25/25),40],[0.25,0.5,0.5,0.25], color="black")
plt.text(50,0.35,'1',fontsize='15')
plt.text(20,0.53,'2',fontsize='15')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

axicon = fig.add_axes([0.2,0.225,0.2,0.3])
plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'$t$ [s]', fontsize=10)
plt.ylabel(r'$A_{\delta}$ [nm]', fontsize=10)
plt.xlim(0.01,4)
plt.ylim(10,3000)
plt.xticks([0.01, 0.1, 1, 4], [0.01, 0.1, 1, 4], fontsize=10)
plt.yticks([10, 100, 1000, 3000], [10, 100, 1000, 3000], fontsize=10)

plt.errorbar(t_amp1_mod, A1_mod, A1_mod_err, fmt='o', solid_capstyle='projecting', capsize=5, marker="v", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h01*(10**6)), mu1*(10**3)))
plt.errorbar(t_amp2_mod, A2_mod, A2_mod_err, fmt='o', solid_capstyle='projecting', capsize=5, marker="o", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h02*(10**6)), mu2*(10**3)))
plt.errorbar(t_amp3_mod, A3_mod, A3_mod_err, fmt='o', solid_capstyle='projecting', capsize=5, marker="^", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h03*(10**6)), mu3*(10**3)))
plt.errorbar(t_amp4_mod, A4_mod, A4_mod_err, fmt='o', solid_capstyle='projecting', capsize=5, marker="v", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h04*(10**6)), mu4*(10**3)))
plt.errorbar(t_amp5_mod, A5_mod, A5_mod_err, fmt='o', solid_capstyle='projecting', capsize=5, marker="o", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h05*(10**6)), mu5*(10**3)))
plt.errorbar(t_amp6_mod, A6_mod, A6_mod_err, fmt='o', solid_capstyle='projecting', capsize=5, marker="^", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h06*(10**6)), mu6*(10**3)))
plt.errorbar(t_amp7_mod, A7_mod, A7_mod_err, fmt='o', solid_capstyle='projecting', capsize=5, marker="v", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h07*(10**6)), mu7*(10**3)))
plt.errorbar(t_amp8_mod, A8_mod, A8_mod_err, fmt='o', solid_capstyle='projecting', capsize=5, marker="o", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h08*(10**6)), mu8*(10**3)))
plt.errorbar(t_amp9_mod, A9_mod, A9_mod_err, fmt='o', solid_capstyle='projecting', capsize=5, marker="^", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h09*(10**6)), mu9*(10**3)))

fig = plt.figure(2)

aa = [(moment_def_1/A1[0])**2,(moment_def_2/A2[0])**2,(moment_def_3/A3[0])**2,(moment_def_4/A4[0])**2,(moment_def_5/A5[0])**2,(moment_def_6/A6[0])**2,(moment_def_7/A7[0])**2,(moment_def_8/A8[0])**2,(moment_def_9/A9[0])**2]
bb = [tc_1/(mu1*h01),tc_2/(mu2*h02),tc_3/(mu3*h03),tc_4/(mu4*h04),tc_5/(mu5*h05),tc_6/(mu6*h06),tc_7/(mu7*h07),tc_8/(mu8*h08),tc_9/(mu9*h09)]

# aa = aa*(10**5)

kk_start = 0
kk_end = 9

plt.plot(aa[kk_start:kk_end],bb[kk_start:kk_end])
plt.scatter(aa[kk_start:kk_end],bb[kk_start:kk_end])

# plt.plot([tc_1,tc_2,tc_3,tc_4,tc_5,tc_6,tc_7,tc_8,tc_9],[moment_def_1,tc_2,tc_3,tc_4,tc_5,tc_6,tc_7,tc_8,tc_9])
# plt.scatter([tc_1,tc_2,tc_3,tc_4,tc_5,tc_6,tc_7,tc_8,tc_9],[tc_1,tc_2,tc_3,tc_4,tc_5,tc_6,tc_7,tc_8,tc_9])

plt.show()

################################################################################

os.chdir('/home/devici/Desktop/amplitude_scaling')

np.savetxt('00050cs0005mum_scaled_time.txt',(t_amp1_mod[set_pt+1:-1] - t_amp1_mod[set_pt])/tc_1)
np.savetxt('00050cs0010mum_scaled_time.txt',(t_amp2_mod[set_pt+1:-1] - t_amp2_mod[set_pt])/tc_2)
np.savetxt('00050cs0015mum_scaled_time.txt',(t_amp3_mod[set_pt+1:-1] - t_amp3_mod[set_pt])/tc_3)
np.savetxt('00100cs0005mum_scaled_time.txt',(t_amp4_mod[set_pt+1:-1] - t_amp4_mod[set_pt])/tc_4)
np.savetxt('00100cs0010mum_scaled_time.txt',(t_amp5_mod[set_pt+1:-1] - t_amp5_mod[set_pt])/tc_5)
np.savetxt('00100cs0015mum_scaled_time.txt',(t_amp6_mod[set_pt+1:-1] - t_amp6_mod[set_pt])/tc_6)
np.savetxt('00200cs0005mum_scaled_time.txt',(t_amp7_mod[set_pt+1:-1] - t_amp7_mod[set_pt])/tc_7)
np.savetxt('00200cs0010mum_scaled_time.txt',(t_amp8_mod[set_pt+1:-1] - t_amp8_mod[set_pt])/tc_8)
np.savetxt('00200cs0015mum_scaled_time.txt',(t_amp9_mod[set_pt+1:-1] - t_amp9_mod[set_pt])/tc_9)

np.savetxt('00050cs0005mum_scaled_amp.txt',(A1_mod[set_pt+1:-1]*(10**-9))/((A1_mod[set_pt])*(10**-9)))
np.savetxt('00050cs0010mum_scaled_amp.txt',(A2_mod[set_pt+1:-1]*(10**-9))/((A2_mod[set_pt])*(10**-9)))
np.savetxt('00050cs0015mum_scaled_amp.txt',(A3_mod[set_pt+1:-1]*(10**-9))/((A3_mod[set_pt])*(10**-9)))
np.savetxt('00100cs0005mum_scaled_amp.txt',(A4_mod[set_pt+1:-1]*(10**-9))/((A4_mod[set_pt])*(10**-9)))
np.savetxt('00100cs0010mum_scaled_amp.txt',(A5_mod[set_pt+1:-1]*(10**-9))/((A5_mod[set_pt])*(10**-9)))
np.savetxt('00100cs0015mum_scaled_amp.txt',(A6_mod[set_pt+1:-1]*(10**-9))/((A6_mod[set_pt])*(10**-9)))
np.savetxt('00200cs0005mum_scaled_amp.txt',(A7_mod[set_pt+1:-1]*(10**-9))/((A7_mod[set_pt])*(10**-9)))
np.savetxt('00200cs0010mum_scaled_amp.txt',(A8_mod[set_pt+1:-1]*(10**-9))/((A8_mod[set_pt])*(10**-9)))
np.savetxt('00200cs0015mum_scaled_amp.txt',(A9_mod[set_pt+1:-1]*(10**-9))/((A9_mod[set_pt])*(10**-9)))

np.savetxt('00050cs0005mum_scaled_errorbar.txt',(A1_mod_err[set_pt+1:-1]*(10**-9))/((A1_mod[set_pt])*(10**-9)))
np.savetxt('00050cs0010mum_scaled_errorbar.txt',(A2_mod_err[set_pt+1:-1]*(10**-9))/((A2_mod[set_pt])*(10**-9)))
np.savetxt('00050cs0015mum_scaled_errorbar.txt',(A3_mod_err[set_pt+1:-1]*(10**-9))/((A3_mod[set_pt])*(10**-9)))
np.savetxt('00100cs0005mum_scaled_errorbar.txt',(A4_mod_err[set_pt+1:-1]*(10**-9))/((A4_mod[set_pt])*(10**-9)))
np.savetxt('00100cs0010mum_scaled_errorbar.txt',(A5_mod_err[set_pt+1:-1]*(10**-9))/((A5_mod[set_pt])*(10**-9)))
np.savetxt('00100cs0015mum_scaled_errorbar.txt',(A6_mod_err[set_pt+1:-1]*(10**-9))/((A6_mod[set_pt])*(10**-9)))
np.savetxt('00200cs0005mum_scaled_errorbar.txt',(A7_mod_err[set_pt+1:-1]*(10**-9))/((A7_mod[set_pt])*(10**-9)))
np.savetxt('00200cs0010mum_scaled_errorbar.txt',(A8_mod_err[set_pt+1:-1]*(10**-9))/((A8_mod[set_pt])*(10**-9)))
np.savetxt('00200cs0015mum_scaled_errorbar.txt',(A9_mod_err[set_pt+1:-1]*(10**-9))/((A9_mod[set_pt])*(10**-9)))

np.savetxt('00050cs0005mum_time.txt',t_amp1_mod)
np.savetxt('00050cs0010mum_time.txt',t_amp2_mod)
np.savetxt('00050cs0015mum_time.txt',t_amp3_mod)
np.savetxt('00100cs0005mum_time.txt',t_amp4_mod)
np.savetxt('00100cs0010mum_time.txt',t_amp5_mod)
np.savetxt('00100cs0015mum_time.txt',t_amp6_mod)
np.savetxt('00200cs0005mum_time.txt',t_amp7_mod)
np.savetxt('00200cs0010mum_time.txt',t_amp8_mod)
np.savetxt('00200cs0015mum_time.txt',t_amp9_mod)

np.savetxt('00050cs0005mum_amp.txt',A1_mod)
np.savetxt('00050cs0010mum_amp.txt',A2_mod)
np.savetxt('00050cs0015mum_amp.txt',A3_mod)
np.savetxt('00100cs0005mum_amp.txt',A4_mod)
np.savetxt('00100cs0010mum_amp.txt',A5_mod)
np.savetxt('00100cs0015mum_amp.txt',A6_mod)
np.savetxt('00200cs0005mum_amp.txt',A7_mod)
np.savetxt('00200cs0010mum_amp.txt',A8_mod)
np.savetxt('00200cs0015mum_amp.txt',A9_mod)

np.savetxt('00050cs0005mum_errorbar.txt',A1_mod_err)
np.savetxt('00050cs0010mum_errorbar.txt',A2_mod_err)
np.savetxt('00050cs0015mum_errorbar.txt',A3_mod_err)
np.savetxt('00100cs0005mum_errorbar.txt',A4_mod_err)
np.savetxt('00100cs0010mum_errorbar.txt',A5_mod_err)
np.savetxt('00100cs0015mum_errorbar.txt',A6_mod_err)
np.savetxt('00200cs0005mum_errorbar.txt',A7_mod_err)
np.savetxt('00200cs0010mum_errorbar.txt',A8_mod_err)
np.savetxt('00200cs0015mum_errorbar.txt',A9_mod_err)
