import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec

conv = 2.967841e-06     #2.5x magnification

gamma = 20/1000

################################################################################
#050 cSt deformation vs height

f1 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00050cs0005mum_r1/2018.09.09 16-53'
os.chdir(f1)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec1 = np.loadtxt('amplitude_decay.txt')
wav_gro1 = np.loadtxt('wavelength_growth.txt')
t_amp1 = amp_dec1[:,0]
A1 = amp_dec1[:,1]
t_wav1 = wav_gro1[:,0]
L1 = wav_gro1[:,1]
mu1 = int(f1[70:75])*(10**-3)
h01 = int(f1[77:81])*(10**-6)

f2 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00050cs0010mum_r1/2018.09.09 17-17'
os.chdir(f2)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec2 = np.loadtxt('amplitude_decay.txt')
wav_gro2 = np.loadtxt('wavelength_growth.txt')
t_amp2 = amp_dec2[:,0]
A2 = amp_dec2[:,1]
t_wav2 = wav_gro2[:,0]
L2 = wav_gro2[:,1]
mu2 = int(f2[70:75])*(10**-3)
h02 = int(f2[77:81])*(10**-6)

f3 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00050cs0015mum_r2/2018.09.09 17-45'
os.chdir(f3)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec3 = np.loadtxt('amplitude_decay.txt')
wav_gro3 = np.loadtxt('wavelength_growth.txt')
t_amp3 = amp_dec3[:,0]
A3 = amp_dec3[:,1]
t_wav3 = wav_gro3[:,0]
L3 = wav_gro3[:,1]
mu3 = int(f3[70:75])*(10**-3)
h03 = int(f3[77:81])*(10**-6)

################################################################################
#100 cSt deformation vs height

f4 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00100cs0005mum_r3/2018.09.09 18-45'
os.chdir(f4)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec4 = np.loadtxt('amplitude_decay.txt')
wav_gro4 = np.loadtxt('wavelength_growth.txt')
t_amp4 = amp_dec4[:,0]
A4 = amp_dec4[:,1]
t_wav4 = wav_gro4[:,0]
L4 = wav_gro4[:,1]
mu4 = int(f4[70:75])*(10**-3)
h04 = int(f4[77:81])*(10**-6)

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
mu5 = int(f5[70:75])*(10**-3)
h05 = int(f5[77:81])*(10**-6)

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
mu6 = int(f6[70:75])*(10**-3)
h06 = int(f6[77:81])*(10**-6)

################################################################################
#200 cSt deformation vs height

f7 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0005mum_r1/2018.09.09 20-17'
os.chdir(f7)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec7 = np.loadtxt('amplitude_decay.txt')
wav_gro7 = np.loadtxt('wavelength_growth.txt')
t_amp7 = amp_dec7[:,0]
A7 = amp_dec7[:,1]
t_wav7 = wav_gro7[:,0]
L7 = wav_gro7[:,1]
mu7 = int(f7[70:75])*(10**-3)
h07 = int(f7[77:81])*(10**-6)

f8 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0010mum_r1/2018.09.09 20-34'
os.chdir(f8)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec8 = np.loadtxt('amplitude_decay.txt')
wav_gro8 = np.loadtxt('wavelength_growth.txt')
t_amp8 = amp_dec8[:,0]
A8 = amp_dec8[:,1]
t_wav8 = wav_gro8[:,0]
L8 = wav_gro8[:,1]
mu8 = int(f8[70:75])*(10**-3)
h08 = int(f8[77:81])*(10**-6)

f9 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0015mum_r1/2018.09.09 20-47'
os.chdir(f9)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec9 = np.loadtxt('amplitude_decay.txt')
wav_gro9 = np.loadtxt('wavelength_growth.txt')
t_amp9 = amp_dec9[:,0]
A9 = amp_dec9[:,1]
t_wav9 = wav_gro9[:,0]
L9 = wav_gro9[:,1]
mu9 = int(f9[70:75])*(10**-3)
h09 = int(f9[77:81])*(10**-6)

################################################################################
# #dimensionalized amplitude variation with time
# gs = gridspec.GridSpec(2, 7)
#
# plt.subplot(gs[0, 0:3])
# ax = plt.gca()
# plt.text(0.2, 6000, '050 mPa.s', fontsize=15)
# ax.set_yscale('log')
# ax.set_xscale('log')
# plt.xlim(0.01,10)
# plt.ylim(10,10000)
# plt.xlabel(r'$t$  $[s]$', fontsize=15)
# plt.ylabel(r'$A_{\delta}$ $[nm]$', fontsize=15)
# ax.scatter(t_amp1, A1, label='05 $\mu m$')
# ax.scatter(t_amp2, A2, label='10 $\mu m$')
# ax.scatter(t_amp3, A3, label='15 $\mu m$')
# handles,labels = ax.get_legend_handles_labels()
# handles = [handles[0], handles[1], handles[2]]
# labels = [labels[0], labels[1], labels[2]]
# ax.legend(handles,labels, loc=1, fontsize=12.5)
# plt.tick_params(axis='both', which='major', labelsize=12.5)
# plt.tick_params(axis='both', which='minor', labelsize=12.5)
#
# plt.subplot(gs[0, 4:7])
# ax = plt.gca()
# plt.text(0.2, 6000, '100 mPa.s', fontsize=15)
# ax.set_yscale('log')
# ax.set_xscale('log')
# plt.xlim(0.01,10)
# plt.ylim(10,10000)
# plt.xlabel(r'$t$  $[s]$', fontsize=15)
# plt.ylabel(r'$A_{\delta}$ $[nm]$', fontsize=15)
# ax.scatter(t_amp4, A4, label='05 $\mu m$')
# ax.scatter(t_amp5, A5, label='10 $\mu m$')
# ax.scatter(t_amp6, A6, label='15 $\mu m$')
# handles,labels = ax.get_legend_handles_labels()
# handles = [handles[0], handles[1], handles[2]]
# labels = [labels[0], labels[1], labels[2]]
# ax.legend(handles,labels, loc=1, fontsize=12.5)
# plt.tick_params(axis='both', which='major', labelsize=12.5)
# plt.tick_params(axis='both', which='minor', labelsize=12.5)
#
# plt.subplot(gs[1, 2:5])
# ax = plt.gca()
# plt.text(0.2, 6000, '200 mPa.s', fontsize=15)
# ax.set_yscale('log')
# ax.set_xscale('log')
# plt.xlim(0.01,10)
# plt.ylim(10,10000)
# plt.xlabel(r'$t$  $[s]$', fontsize=15)
# plt.ylabel(r'$A_{\delta}$ $[nm]$', fontsize=15)
# ax.scatter(t_amp7, A7, label='05 $\mu m$')
# ax.scatter(t_amp8, A8, label='10 $\mu m$')
# ax.scatter(t_amp9, A9, label='15 $\mu m$')
# handles,labels = ax.get_legend_handles_labels()
# handles = [handles[0], handles[1], handles[2]]
# labels = [labels[0], labels[1], labels[2]]
# ax.legend(handles,labels, loc=1, fontsize=12.5)
# plt.tick_params(axis='both', which='major', labelsize=12.5)
# plt.tick_params(axis='both', which='minor', labelsize=12.5)
#
# plt.show()

################################################################################
#dimensionalized wavelength variation with time
fig, ax = plt.subplots(2,2)

plt.subplot(2,2,1)
ax = plt.gca()
plt.title('050 cSt')
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlim(0.01,10)
plt.ylim(10,10000)
plt.xlabel(r'$t$ $[s]$')
plt.ylabel(r'$L_{\lambda}$ $[nm]$')
ax.scatter(t_amp1, L1, label='05 $\mu m$')
ax.scatter(t_amp2, L2, label='10 $\mu m$')
ax.scatter(t_amp3, L3, label='15 $\mu m$')
handles,labels = ax.get_legend_handles_labels()
handles = [handles[0], handles[1], handles[2]]
labels = [labels[0], labels[1], labels[2]]
ax.legend(handles,labels)
#ax.grid(True,which='both',ls='-')
# plt.savefig('amplitude_decay.png')

plt.subplot(2,2,2)
ax = plt.gca()
plt.title('100 cSt')
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlim(0.01,10)
plt.ylim(10,10000)
plt.xlabel(r'$t$ $[s]$')
plt.ylabel(r'$L_{\lambda}$ $[nm]$')
ax.scatter(t_amp4, L4, label='05 $\mu m$')
ax.scatter(t_amp5, L5, label='10 $\mu m$')
ax.scatter(t_amp6, L6, label='15 $\mu m$')
handles,labels = ax.get_legend_handles_labels()
handles = [handles[0], handles[1], handles[2]]
labels = [labels[0], labels[1], labels[2]]
ax.legend(handles,labels)
#ax.grid(True,which='both',ls='-')
# plt.savefig('amplitude_decay.png')

plt.subplot(2,2,3)
ax = plt.gca()
plt.title('200 cSt')
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlim(0.01,10)
plt.ylim(10,10000)
plt.xlabel(r'$t$ $[s]$')
plt.ylabel(r'$L_{\lambda}$ $[nm]$')
ax.scatter(t_amp7, L7, label='05 $\mu m$')
ax.scatter(t_amp8, L8, label='10 $\mu m$')
ax.scatter(t_amp9, L9, label='15 $\mu m$')
handles,labels = ax.get_legend_handles_labels()
handles = [handles[0], handles[1], handles[2]]
labels = [labels[0], labels[1], labels[2]]
ax.legend(handles,labels)
#ax.grid(True,which='both',ls='-')
# plt.savefig('amplitude_decay.png')

plt.subplot(2,2,4)
ax = plt.gca()
plt.title('350 cSt')
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlim(0.01,10)
plt.ylim(10,10000)
plt.xlabel(r'$t$ $[s]$')
plt.ylabel(r'$L_{\lambda}$ $[nm]$')
# ax.scatter(t_amp7, L7, label='05 $\mu m$')
# ax.scatter(t_amp8, L8, label='10 $\mu m$')
# ax.scatter(t_amp9, L9, label='15 $\mu m$')
# handles,labels = ax.get_legend_handles_labels()
# handles = [handles[0], handles[1], handles[2]]
# labels = [labels[0], labels[1], labels[2]]
# ax.legend(handles,labels)
#ax.grid(True,which='both',ls='-')
# plt.savefig('amplitude_decay.png')

plt.show()

################################################################################
