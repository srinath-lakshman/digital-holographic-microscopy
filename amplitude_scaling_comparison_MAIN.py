import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
from FUNC_ import power_law_fit
from FUNC_ import sphere_to_pancake
from FUNC_ import binning_data

conv = 2.967841e-06                                                             #2.5x magnification

sigma = 20.3/1000

################################################################################

# f1 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00050cs0005mum_r1/2018.09.09 16-53'
f1 = r'F:\srinath_dhm\impact_over_thin_films\speed1\00050cs0005mum_r2\2018.09.09 16-59'
# f1 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00050cs0005mum_r3/2018.09.09 17-07'
os.chdir(f1)
os.chdir(os.getcwd() + r'\info')
os.chdir(os.getcwd() + r'\relaxation_profiles')
amp_dec1 = np.loadtxt('amplitude_decay.txt')
wav_gro1 = np.loadtxt('wavelength_growth.txt')
t_amp1 = amp_dec1[:,0]
A1 = amp_dec1[:,1]
t_wav1 = wav_gro1[:,0]
L1 = wav_gro1[:,1]
# mu1 = int(f1[63:68])
# h01 = int(f1[70:74])
mu1 = 50*(10**-3)
h01 = 5*(10**-6)

plt.figure()
plt.scatter(t_amp1,A1)
plt.show()

# A1_mod = binning_data(A1,int(np.math.ceil(len(A1)*(1/200))*2))

# f2 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00050cs0010mum_r1/2018.09.09 17-17'
# # f2 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00050cs0010mum_r2_BAD/2018.09.09 17-23'
# # f2 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00050cs0010mum_r3*/2018.09.09 17-29'
# os.chdir(f2)
# os.chdir(os.getcwd() + r'/info')
# os.chdir(os.getcwd() + r'/relaxation_profiles')
# amp_dec2 = np.loadtxt('amplitude_decay.txt')
# wav_gro2 = np.loadtxt('wavelength_growth.txt')
# t_amp2 = amp_dec2[:,0]
# A2 = amp_dec2[:,1]
# t_wav2 = wav_gro2[:,0]
# L2 = wav_gro2[:,1]
# # mu2 = int(f2[63:68])
# # h02 = int(f2[70:74])
# mu2 = 50*(10**-3)
# h02 = 10*(10**-6)

# f3 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00050cs0015mum_r1*/2018.09.09 17-39'
f3 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00050cs0015mum_r2*/2018.09.09 17-45'
# f3 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00050cs0015mum_r3*/2018.09.09 17-51'
os.chdir(f3)
os.chdir(os.getcwd() + r'\info')
os.chdir(os.getcwd() + r'\relaxation_profiles')
amp_dec3 = np.loadtxt('amplitude_decay.txt')
wav_gro3 = np.loadtxt('wavelength_growth.txt')
t_amp3 = amp_dec3[:,0]
A3 = amp_dec3[:,1]
t_wav3 = wav_gro3[:,0]
L3 = wav_gro3[:,1]
# mu3 = int(f3[63:68])
# h03 = int(f3[70:74])
mu3 = 50*(10**-3)
h03 = 15*(10**-6)

################################################################################
#100 cSt deformation vs height

# f4 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00100cs0005mum_r1/2018.09.09 18-39'
f4 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00100cs0005mum_r2/2018.09.09 18-43'
# f4 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00100cs0005mum_r3/2018.09.09 18-45'
os.chdir(f4)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec4 = np.loadtxt('amplitude_decay.txt')
wav_gro4 = np.loadtxt('wavelength_growth.txt')
t_amp4 = amp_dec4[:,0]
A4 = amp_dec4[:,1]
t_wav4 = wav_gro4[:,0]
L4 = wav_gro4[:,1]
# mu4 = int(f4[63:68])
# h04 = int(f4[70:74])
mu4 = 100*(10**-3)
h04 = 5*(10**-6)

f5 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00100cs0010mum_r1/2018.09.09 18-53'
os.chdir(f5)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec5 = np.loadtxt('amplitude_decay.txt')
wav_gro5 = np.loadtxt('wavelength_growth.txt')
t_amp5 = amp_dec5[:,0]
A5 = amp_dec5[:,1]
t_wav5 = wav_gro5[:,0]
L5 = wav_gro5[:,1]
# mu5 = int(f5[63:68])
# h05 = int(f5[70:74])
mu5 = 100*(10**-3)
h05 = 10*(10**-6)

f6 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00100cs0015mum_r3/2018.09.09 19-24'
os.chdir(f6)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec6 = np.loadtxt('amplitude_decay.txt')
wav_gro6 = np.loadtxt('wavelength_growth.txt')
t_amp6 = amp_dec6[:,0]
A6 = amp_dec6[:,1]
t_wav6 = wav_gro6[:,0]
L6 = wav_gro6[:,1]
# mu6 = int(f6[63:68])
# h06 = int(f6[70:74])
mu6 = 100*(10**-3)
h06 = 15*(10**-6)

################################################################################
#200 cSt deformation vs height

f7 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0005mum_r1/2018.09.09 20-17'
# f7 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0005mum_r2/2018.09.09 20-20'
# f7 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0005mum_r3/2018.09.09 20-27'
os.chdir(f7)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec7 = np.loadtxt('amplitude_decay.txt')
wav_gro7 = np.loadtxt('wavelength_growth.txt')
t_amp7 = amp_dec7[:,0]
A7 = amp_dec7[:,1]
t_wav7 = wav_gro7[:,0]
L7 = wav_gro7[:,1]
# mu7 = int(f7[63:68])
# h07 = int(f7[70:74])
mu7 = 200*(10**-3)
h07 = 5*(10**-6)

f8 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0010mum_r1_GOOD/2018.09.09 20-34'
os.chdir(f8)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec8 = np.loadtxt('amplitude_decay.txt')
wav_gro8 = np.loadtxt('wavelength_growth.txt')
t_amp8 = amp_dec8[:,0]
A8 = amp_dec8[:,1]
t_wav8 = wav_gro8[:,0]
L8 = wav_gro8[:,1]
# mu8 = int(f8[63:68])
# h08 = int(f8[70:74])
mu8 = 200*(10**-3)
h08 = 10*(10**-6)

f9 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0015mum_r1_GOOD/2018.09.09 20-47'
os.chdir(f9)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec9 = np.loadtxt('amplitude_decay.txt')
wav_gro9 = np.loadtxt('wavelength_growth.txt')
t_amp9 = amp_dec9[:,0]
A9 = amp_dec9[:,1]
t_wav9 = wav_gro9[:,0]
L9 = wav_gro9[:,1]
# mu9 = int(f9[63:68])
# h09 = int(f9[70:74])
mu9 = 200*(10**-3)
h09 = 15*(10**-6)

################################################################################

plt.figure(1)
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')

plt.xlim(10,1000000)
plt.ylim(0.1,1)
plt.xticks([10, 100, 1000, 10000, 100000, 1000000], fontsize=12.5)
plt.yticks([0.1, 1], fontsize=12.5)

plt.xlabel(r'$\left( t - t_{0} \right)/\left( \frac{\eta h_{0}}{\gamma} \right)$', fontsize=25)
plt.ylabel(r'$\frac{A_{\delta}}{A_{\delta,0}}$', fontsize=25)
plt.scatter((t_amp1-t_amp1[0])/((h01*mu1)/sigma), (A1*(10**-9))/(max(A1)*(10**-9)), marker=".", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h01*(10**6)), mu1*(10**3)))
# plt.scatter((t_amp2-t_amp2[0])/((h02*mu2)/sigma), (A2*(10**-9))/(max(A2)*(10**-9)), marker="+", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h02*(10**6)), mu2*(10**3)))
# plt.scatter((t_amp3-t_amp3[0])/((h03*mu3)/sigma), (A3*(10**-9))/(max(A3)*(10**-9)), marker="x", color="#1f77b4", label=r"%02i $\mu m$, %03i mPa.s" %(round(h03*(10**6)), mu3*(10**3)))
# plt.scatter((t_amp4-t_amp4[0])/((h04*mu4)/sigma), (A4*(10**-9))/(max(A4)*(10**-9)), marker=".", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h04*(10**6)), mu4*(10**3)))
# plt.scatter((t_amp5-t_amp5[0])/((h05*mu5)/sigma), (A5*(10**-9))/(max(A5)*(10**-9)), marker="+", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h05*(10**6)), mu5*(10**3)))
# plt.scatter((t_amp6-t_amp6[0])/((h06*mu6)/sigma), (A6*(10**-9))/(max(A6)*(10**-9)), marker="x", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h06*(10**6)), mu6*(10**3)))
# plt.scatter((t_amp7-t_amp7[0])/((h07*mu7)/sigma), (A7*(10**-9))/(max(A7)*(10**-9)), marker=".", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h07*(10**6)), mu7*(10**3)))
# plt.scatter((t_amp8-t_amp8[0])/((h08*mu8)/sigma), (A8*(10**-9))/(max(A8)*(10**-9)), marker="+", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h08*(10**6)), mu8*(10**3)))
# plt.scatter((t_amp9-t_amp7[0])/((h09*mu9)/sigma), (A9*(10**-9))/(max(A9)*(10**-9)), marker="x", color="#2ca02c", label=r"%02i $\mu m$, %03i mPa.s" %(round(h09*(10**6)), mu9*(10**3)))

plt.scatter((t_amp1-t_amp1[0])/((h01*mu1)/sigma), (A1_mod*(10**-9))/(max(A1)*(10**-9)), marker=".", color="#ff7f0e", label=r"%02i $\mu m$, %03i mPa.s" %(round(h01*(10**6)), mu1*(10**3)))

plt.plot([100000,100000*((4**2)/(2**2)),100000*((4**2)/(2**2)),100000],[0.4,0.4,0.2,0.4], color="black")
plt.text(500000,0.3,'1',fontsize='20')
plt.text(200000,0.425,'2',fontsize='20')

ax.legend(fontsize=12.5,loc=3)
plt.show()

################################################################################
