import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
from FUNC_ import power_law_fit
from FUNC_ import sphere_to_pancake
from scipy import stats

conv = 2.967841e-06                                                             #2.5x magnification

sigma = 20.3/1000

################################################################################
#050 cSt deformation vs height

f1 = r'F:\srinath_dhm\impact_over_thin_films\speed1\00050cs0005mum_r1\2018.09.09 16-53'
os.chdir(f1)
os.chdir(os.getcwd() + r'\info')
os.chdir(os.getcwd() + r'\relaxation_profiles')
amp_dec1 = np.loadtxt('amplitude_decay.txt')
wav_gro1 = np.loadtxt('wavelength_growth.txt')
t_amp1 = amp_dec1[:,0]
A1 = amp_dec1[:,1]
t_wav1 = wav_gro1[:,0]
L1 = wav_gro1[:,1]
initial_amp1, power_amp1, A_fit1 = power_law_fit(t_amp1, A1)
initial_wav1, power_wav1, L_fit1 = power_law_fit(t_wav1, L1)
# mu1 = int(f1[63:68])
# h01 = int(f1[70:74])
mu1 = 50*(10**-3)
h01 = 5*(10**-6)

# nnn = 100
#
# bin_means, bin_edges, binnumber = stats.binned_statistic(t_wav1, L1, 'mean', bins=nnn)
# bin_means_1, bin_edges_1, binnumber_1 = stats.binned_statistic(t_wav1, L1, 'std', bins=nnn)
#
# new_t = np.diff(bin_edges)/2 + bin_edges[0:-1]
#
# plt.errorbar(new_t,bin_means,bin_means_1,color='black')
# plt.scatter(t_wav1,L1)
# plt.show()
# input()

# plt.scatter(t_wav1,L1)
# plt.plot(new_t,bin_means)
# plt.scatter(new_t,bin_means)
# plt.show()
# input()


f2 = r'F:\srinath_dhm\impact_over_thin_films\speed1\00050cs0010mum_r1\2018.09.09 17-17'
os.chdir(f2)
os.chdir(os.getcwd() + r'\info')
os.chdir(os.getcwd() + r'\relaxation_profiles')
amp_dec2 = np.loadtxt('amplitude_decay.txt')
wav_gro2 = np.loadtxt('wavelength_growth.txt')
t_amp2 = amp_dec2[:,0]
A2 = amp_dec2[:,1]
t_wav2 = wav_gro2[:,0]
L2 = wav_gro2[:,1]
initial_amp2, power_amp2, A_fit2 = power_law_fit(t_amp2, A2)
initial_wav2, power_wav2, L_fit2 = power_law_fit(t_wav2, L2)
# mu2 = int(f2[63:68])
# h02 = int(f2[70:74])
mu2 = 50*(10**-3)
h02 = 10*(10**-6)

f3 = r'F:\srinath_dhm\impact_over_thin_films\speed1\00050cs0015mum_r2\2018.09.09 17-45'
os.chdir(f3)
os.chdir(os.getcwd() + r'\info')
os.chdir(os.getcwd() + r'\relaxation_profiles')
amp_dec3 = np.loadtxt('amplitude_decay.txt')
wav_gro3 = np.loadtxt('wavelength_growth.txt')
t_amp3 = amp_dec3[:,0]
A3 = amp_dec3[:,1]
t_wav3 = wav_gro3[0:500,0]
L3 = wav_gro3[0:500,1]
initial_amp3, power_amp3, A_fit3 = power_law_fit(t_amp3, A3)
initial_wav3, power_wav3, L_fit3 = power_law_fit(t_wav3, L3)
# mu3 = int(f3[63:68])
# h03 = int(f3[70:74])
mu3 = 50*(10**-3)
h03 = 15*(10**-6)

################################################################################
#100 cSt deformation vs height

f4 = r'F:\srinath_dhm\impact_over_thin_films\speed1\00100cs0005mum_r2\2018.09.09 18-43'
os.chdir(f4)
os.chdir(os.getcwd() + r'\info')
os.chdir(os.getcwd() + r'\relaxation_profiles')
amp_dec4 = np.loadtxt('amplitude_decay.txt')
wav_gro4 = np.loadtxt('wavelength_growth.txt')
t_amp4 = amp_dec4[:,0]
A4 = amp_dec4[:,1]
t_wav4 = wav_gro4[:,0]
L4 = wav_gro4[:,1]
initial_amp4, power_amp4, A_fit4 = power_law_fit(t_amp4, A4)
initial_wav4, power_wav4, L_fit4 = power_law_fit(t_wav4, L4)
# mu4 = int(f4[63:68])
# h04 = int(f4[70:74])
mu4 = 100*(10**-3)
h04 = 5*(10**-6)

f5 = r'F:\srinath_dhm\impact_over_thin_films\speed1\00100cs0010mum_r1\2018.09.09 18-53'
os.chdir(f5)
os.chdir(os.getcwd() + r'\info')
os.chdir(os.getcwd() + r'\relaxation_profiles')
amp_dec5 = np.loadtxt('amplitude_decay.txt')
wav_gro5 = np.loadtxt('wavelength_growth.txt')
t_amp5 = amp_dec5[:,0]
A5 = amp_dec5[:,1]
t_wav5 = wav_gro5[:,0]
L5 = wav_gro5[:,1]
initial_amp5, power_amp5, A_fit5 = power_law_fit(t_amp5, A5)
initial_wav5, power_wav5, L_fit5 = power_law_fit(t_wav5, L5)
# mu5 = int(f5[63:68])
# h05 = int(f5[70:74])
mu5 = 100*(10**-3)
h05 = 10*(10**-6)

f6 = r'F:\srinath_dhm\impact_over_thin_films\speed1\00100cs0015mum_r3\2018.09.09 19-24'
os.chdir(f6)
os.chdir(os.getcwd() + r'\info')
os.chdir(os.getcwd() + r'\relaxation_profiles')
amp_dec6 = np.loadtxt('amplitude_decay.txt')
wav_gro6 = np.loadtxt('wavelength_growth.txt')
t_amp6 = amp_dec6[:,0]
A6 = amp_dec6[:,1]
t_wav6 = wav_gro6[0:800,0]
L6 = wav_gro6[0:800,1]
initial_amp6, power_amp6, A_fit6 = power_law_fit(t_amp6, A6)
initial_wav6, power_wav6, L_fit6 = power_law_fit(t_wav6, L6)
# mu6 = int(f6[63:68])
# h06 = int(f6[70:74])
mu6 = 100*(10**-3)
h06 = 15*(10**-6)

################################################################################
#200 cSt deformation vs height

f7 = r'F:\srinath_dhm\impact_over_thin_films\speed1\00200cs0005mum_r3\2018.09.09 20-27'
os.chdir(f7)
os.chdir(os.getcwd() + r'\info')
os.chdir(os.getcwd() + r'\relaxation_profiles')
amp_dec7 = np.loadtxt('amplitude_decay.txt')
wav_gro7 = np.loadtxt('wavelength_growth.txt')
t_amp7 = amp_dec7[:,0]
A7 = amp_dec7[:,1]
t_wav7 = wav_gro7[:,0]
L7 = wav_gro7[:,1]
initial_amp7, power_amp7, A_fit7 = power_law_fit(t_amp7, A7)
initial_wav7, power_wav7, L_fit7 = power_law_fit(t_wav7, L7)
# mu7 = int(f7[63:68])
# h07 = int(f7[70:74])
mu7 = 200*(10**-3)
h07 = 5*(10**-6)

f8 = r'F:\srinath_dhm\impact_over_thin_films\speed1\00200cs0010mum_r3_TODO\2018.09.09 20-40'
os.chdir(f8)
os.chdir(os.getcwd() + r'\info')
os.chdir(os.getcwd() + r'\relaxation_profiles')
amp_dec8 = np.loadtxt('amplitude_decay.txt')
wav_gro8 = np.loadtxt('wavelength_growth.txt')
t_amp8 = amp_dec8[:,0]
A8 = amp_dec8[:,1]
t_wav8 = wav_gro8[:,0]
L8 = wav_gro8[:,1]
initial_amp8, power_amp8, A_fit8 = power_law_fit(t_amp8, A8)
initial_wav8, power_wav8, L_fit8 = power_law_fit(t_wav8, L8)
# mu8 = int(f8[63:68])
# h08 = int(f8[70:74])
mu8 = 200*(10**-3)
h08 = 10*(10**-6)

f9 = r'F:\srinath_dhm\impact_over_thin_films\speed1\00200cs0015mum_r1_GOOD\2018.09.09 20-47'
os.chdir(f9)
os.chdir(os.getcwd() + r'\info')
os.chdir(os.getcwd() + r'\relaxation_profiles')
amp_dec9 = np.loadtxt('amplitude_decay.txt')
wav_gro9 = np.loadtxt('wavelength_growth.txt')
t_amp9 = amp_dec9[:,0]
A9 = amp_dec9[:,1]
t_wav9 = wav_gro9[:,0]
L9 = wav_gro9[:,1]
initial_amp9, power_amp9, A_fit9 = power_law_fit(t_amp9, A9)
initial_wav9, power_wav9, L_fit9 = power_law_fit(t_wav9, L9)
# mu9 = int(f9[63:68])
# h09 = int(f9[70:74])
mu9 = 200*(10**-3)
h09 = 15*(10**-6)

################################################################################

x_coor = np.concatenate(((t_wav1)/((h01*mu1)/sigma),(t_wav2)/((h02*mu2)/sigma),(t_wav3)/((h03*mu3)/sigma), \
                          (t_wav4)/((h04*mu4)/sigma),(t_wav5)/((h05*mu5)/sigma),(t_wav6)/((h06*mu6)/sigma), \
                          (t_wav7)/((h07*mu7)/sigma),(t_wav8)/((h08*mu8)/sigma),(t_wav9)/((h09*mu9)/sigma)))
y_coor = np.concatenate(((L1*(10**-6))/h01,(L2*(10**-6))/h02,(L3*(10**-6))/h03, \
                          (L4*(10**-6))/h04,(L5*(10**-6))/h05,(L6*(10**-6))/h06, \
                          (L7*(10**-6))/h07,(L8*(10**-6))/h08,(L9*(10**-6))/h09))

initial_coor, power_coor, y_fit_coor = power_law_fit(x_coor,y_coor)

################################################################################

plt.figure()
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')

plt.xlabel(r'$t/\left( \frac{\eta h_{0}}{\gamma} \right)$', fontsize=25)
plt.ylabel(r'$\frac{L_{\lambda}}{h_{0}}$', fontsize=25)
# plt.xlim(50,500000)
# plt.xticks([50, 100, 1000, 10000, 100000, 500000], fontsize=12.5)

plt.xlim(50,500000)
plt.xticks([50, 100, 1000, 10000, 100000, 500000], fontsize=12.5)
plt.ylim(1, 100)
plt.yticks([1, 10, 100], fontsize=12.5)

plt.plot([1, (100/initial_coor)**(1/power_coor)],[initial_coor, 100],'--',color='black')

# plt.ylim(5, 100)
# plt.yticks([5, 10, 100], fontsize=12.5)
plt.scatter((t_wav1)/((h01*mu1)/sigma), (L1*(10**-6))/h01, marker=".", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h01*(10**6)), mu1*(10**3)))
plt.scatter((t_wav2)/((h02*mu2)/sigma), (L2*(10**-6))/h02, marker="+", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h02*(10**6)), mu2*(10**3)))
plt.scatter((t_wav3)/((h03*mu3)/sigma), (L3*(10**-6))/h03, marker="x", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h03*(10**6)), mu3*(10**3)))
plt.scatter((t_wav4)/((h04*mu4)/sigma), (L4*(10**-6))/h04, marker=".", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h04*(10**6)), mu4*(10**3)))
plt.scatter((t_wav5)/((h05*mu5)/sigma), (L5*(10**-6))/h05, marker="+", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h05*(10**6)), mu5*(10**3)))
plt.scatter((t_wav6)/((h06*mu6)/sigma), (L6*(10**-6))/h06, marker="x", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h06*(10**6)), mu6*(10**3)))
plt.scatter((t_wav7)/((h07*mu7)/sigma), (L7*(10**-6))/h07, marker=".", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h07*(10**6)), mu7*(10**3)))
plt.scatter((t_wav8)/((h08*mu8)/sigma), (L8*(10**-6))/h08, marker="+", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h08*(10**6)), mu8*(10**3)))
plt.scatter((t_wav9)/((h09*mu9)/sigma), (L9*(10**-6))/h09, marker="x", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h09*(10**6)), mu9*(10**3)))

plt.plot([1000,100000,100000,1000],[10,10*np.sqrt(10),10,10], color="black")
plt.text(10000,8.5,'4',fontsize='20')
plt.text(115000,16,'1',fontsize='20')

ax.legend(fontsize=12.5,loc=2)

# plt.figure(2)

# plt.scatter(x_coor,y_coor,marker='x')
# plt.plot(x_coor,y_fit_coor,marker='x',color='black')
plt.show()

################################################################################

################################################################################

# plt.figure()
# ax = plt.gca()
# ax.set_yscale('log')
# ax.set_xscale('log')
#
# plt.xlabel(r'$t/\left( \frac{\eta h_{0}}{\gamma} \right)$', fontsize=25)
# plt.ylabel(r'$\frac{L_{\lambda}}{h_{0}}$', fontsize=25)
# plt.xlim(50,500000)
# plt.xticks([50, 100, 1000, 10000, 100000, 500000], fontsize=12.5)
# plt.ylim(5, 100)
# plt.yticks([5, 10, 100], fontsize=12.5)
#
# plt.scatter((t_wav1)/((h01*mu1)/sigma), (L1*(10**-6))/h01, marker=".", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h01*(10**6)), mu1*(10**3)))
# plt.scatter((t_wav2)/((h02*mu2)/sigma), (L2*(10**-6))/h02, marker="+", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h02*(10**6)), mu2*(10**3)))
# plt.scatter((t_wav3)/((h03*mu3)/sigma), (L3*(10**-6))/h03, marker="x", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h03*(10**6)), mu3*(10**3)))
# plt.scatter((t_wav4)/((h04*mu4)/sigma), (L4*(10**-6))/h04, marker=".", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h04*(10**6)), mu4*(10**3)))
# plt.scatter((t_wav5)/((h05*mu5)/sigma), (L5*(10**-6))/h05, marker="+", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h05*(10**6)), mu5*(10**3)))
# plt.scatter((t_wav6)/((h06*mu6)/sigma), (L6*(10**-6))/h06, marker="x", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h06*(10**6)), mu6*(10**3)))
# plt.scatter((t_wav7)/((h07*mu7)/sigma), (L7*(10**-6))/h07, marker=".", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h07*(10**6)), mu7*(10**3)))
# plt.scatter((t_wav8)/((h08*mu8)/sigma), (L8*(10**-6))/h08, marker="+", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h08*(10**6)), mu8*(10**3)))
# plt.scatter((t_wav9)/((h09*mu9)/sigma), (L9*(10**-6))/h09, marker="x", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h09*(10**6)), mu9*(10**3)))
#
# plt.plot([1, (100/initial_coor)**(1/power_coor)],[initial_coor, 100],'--',color='black')
#
# plt.plot([1000,100000,100000,1000],[10,10*np.sqrt(10),10,10], color="black")
# plt.text(10000,8.5,'4',fontsize='20')
# plt.text(115000,16,'1',fontsize='20')
#
# ax.legend(fontsize=12.5,loc=2)
#
# plt.show()

################################################################################
