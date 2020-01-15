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
# mu1 = int(f1[63:68])
# h01 = int(f1[70:74])
mu1 = 50
h01 = 5

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
# mu2 = int(f2[63:68])
# h02 = int(f2[70:74])
mu2 = 50
h02 = 10

f3 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00050cs0015mum_r2*/2018.09.09 17-45'
os.chdir(f3)
os.chdir(os.getcwd() + r'/info')
os.chdir(os.getcwd() + r'/relaxation_profiles')
amp_dec3 = np.loadtxt('amplitude_decay.txt')
wav_gro3 = np.loadtxt('wavelength_growth.txt')
t_amp3 = amp_dec3[:,0]
A3 = amp_dec3[:,1]
t_wav3 = wav_gro3[:,0]
L3 = wav_gro3[:,1]
# mu3 = int(f3[63:68])
# h03 = int(f3[70:74])
mu3 = 50
h03 = 15

################################################################################
#100 cSt deformation vs height

f4 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00100cs0005mum_r2/2018.09.09 18-43'
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
mu4 = 100
h04 = 5

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
mu5 = 100
h05 = 10

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
mu6 = 100
h06 = 15

################################################################################
#200 cSt deformation vs height

f7 = r'/media/devici/srinath_dhm02/srinath_dhm/impact_over_thin_films/speed1/00200cs0005mum_r3/2018.09.09 20-27'
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
mu7 = 200
h07 = 5

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
mu8 = 200
h08 = 10

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
mu9 = 200
h09 = 15

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
# non-dimensionalized amplitude variation with non-dimensionalized time
# fig, ax = plt.subplots(2,2)
gs = gridspec.GridSpec(2, 7)
# ax1 = plt.subplot(gs[0, 0:2])
# ax2 = plt.subplot(gs[0,2:])
# ax3 = plt.subplot(gs[1,1:3])
# fig = gcf()
#
# print(t_amp1-t_amp1[0])
# input()

plt.subplot(gs[0, 0:3])
ax = plt.gca()
# ax = plt.subplot(gs[0, 0:2])
# plt.title('050 cSt')
plt.text(1250, 6, r'050 mPa.s', fontsize=15)
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlim(10,1000000)
plt.ylim(0.001,10)
plt.xlabel(r'$t$/$\left( \frac{\mu h_0}{\sigma} \right)$', fontsize=15)
plt.ylabel(r'$A_{\delta}$/$h_{0}$', fontsize=15)
ax.scatter((t_amp1-t_amp1[0])/((mu1*h01)/(gamma*1000000000)), A1/A1[0], label='05 $\mu m$')
ax.scatter((t_amp2-t_amp2[0])/((mu2*h02)/(gamma*1000000000)), A2/A2[0], label='10 $\mu m$')
ax.scatter((t_amp3-t_amp3[0])/((mu3*h03)/(gamma*1000000000)), A3/A3[0], label='15 $\mu m$')
handles,labels = ax.get_legend_handles_labels()
handles = [handles[0], handles[1], handles[2]]
labels = [labels[0], labels[1], labels[2]]
ax.legend(handles,labels, loc=1, fontsize=12.5)
plt.tick_params(axis='both', which='major', labelsize=12.5)
plt.tick_params(axis='both', which='minor', labelsize=12.5)
#ax.grid(True,which='both',ls='-')
# plt.savefig('amplitude_decay.png')

plt.subplot(gs[0, 4:7])
ax = plt.gca()
# ax = plt.subplot(gs[0,2:])
# plt.title('100 cSt')
plt.text(5000, 0.5, r'100 mPa.s', fontsize=15)
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlim(100,1000000)
plt.ylim(0.001,1)
plt.xlabel(r'$t$/$\left( \frac{\mu h_0}{\sigma} \right)$', fontsize=15)
plt.ylabel(r'$A_{\delta}$/$h_{0}$', fontsize=15)
ax.scatter(t_amp4/((mu4*h04)/(gamma*1000000000)), (A4/h04)*(1/1000), label='05 $\mu m$')
ax.scatter(t_amp5/((mu5*h05)/(gamma*1000000000)), (A5/h05)*(1/1000), label='10 $\mu m$')
ax.scatter(t_amp6/((mu6*h06)/(gamma*1000000000)), (A6/h06)*(1/1000), label='15 $\mu m$')
# ax.scatter(t_amp4, A4, label='05 $\mu m$')
# ax.scatter(t_amp5, A5, label='10 $\mu m$')
# ax.scatter(t_amp6, A6, label='15 $\mu m$')
handles,labels = ax.get_legend_handles_labels()
handles = [handles[0], handles[1], handles[2]]
labels = [labels[0], labels[1], labels[2]]
ax.legend(handles,labels, loc=1, fontsize=12.5)
plt.tick_params(axis='both', which='major', labelsize=12.5)
plt.tick_params(axis='both', which='minor', labelsize=12.5)
#ax.grid(True,which='both',ls='-')
# plt.savefig('amplitude_decay.png')

plt.subplot(gs[1, 2:5])
ax = plt.gca()
# ax = plt.subplot(gs[3,1:3])
# plt.title('200 cSt')
plt.text(5000, 0.5, r'200 mPa.s', fontsize=15)
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlim(100,1000000)
plt.ylim(0.001,1)
plt.xlabel(r'$t$/$\left( \frac{\mu h_0}{\sigma} \right)$', fontsize=15)
plt.ylabel(r'$A_{\delta}$/$h_{0}$', fontsize=15)
ax.scatter(t_amp7/((mu7*h07)/(gamma*1000000000)), (A7/h07)*(1/1000), label='05 $\mu m$')
ax.scatter(t_amp8/((mu8*h08)/(gamma*1000000000)), (A8/h08)*(1/1000), label='10 $\mu m$')
ax.scatter(t_amp9/((mu9*h09)/(gamma*1000000000)), (A9/h09)*(1/1000), label='15 $\mu m$')
# ax.scatter(t_amp7, A7, label='05 $\mu m$')
# ax.scatter(t_amp8, A8, label='10 $\mu m$')
# ax.scatter(t_amp9, A9, label='15 $\mu m$')
handles,labels = ax.get_legend_handles_labels()
handles = [handles[0], handles[1], handles[2]]
labels = [labels[0], labels[1], labels[2]]
ax.legend(handles,labels, loc=1, fontsize=12.5)
plt.tick_params(axis='both', which='major', labelsize=12.5)
plt.tick_params(axis='both', which='minor', labelsize=12.5)
ax.grid(True,which='both',ls='-')
plt.savefig('amplitude_decay.png')

plt.subplot(2,2,4)
ax = plt.gca()
# plt.title('350 cSt')
plt.text(5000, 0.5, r'350 mPa.s', fontsize=15)
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlim(100,1000000)
plt.ylim(0.001,1)
plt.xlabel(r'$t$/$\left( \frac{\mu h_0}{\sigma} \right)$', fontsize=15)
plt.ylabel(r'$A_{\delta}$/$h_{0}$', fontsize=15)
# ax.scatter(t_amp7, A7, label='05 $\mu m$')
# ax.scatter(t_amp8, A8, label='10 $\mu m$')
# ax.scatter(t_amp9, A9, label='15 $\mu m$')
# handles,labels = ax.get_legend_handles_labels()
# handles = [handles[0], handles[1], handles[2]]
# labels = [labels[0], labels[1], labels[2]]
# ax.legend(handles,labels)
#ax.grid(True,which='both',ls='-')
# plt.savefig('amplitude_decay.png')

plt.show()

################################################################################
# # dimensionalized amplitude variation with film height
# gs = gridspec.GridSpec(2, 7)
#
# plt.subplot(gs[0, 0:3])
# ax = plt.gca()
# plt.text(0.2, 6000, '05 $\mu m$', fontsize=15)
# ax.set_yscale('log')
# ax.set_xscale('log')
# plt.xlim(0.01,10)
# plt.ylim(10,10000)
# plt.xlabel(r'$t$  $[s]$', fontsize=15)
# plt.ylabel(r'$A_{\delta}$ $[nm]$', fontsize=15)
# ax.scatter(t_amp1, A1, label='050 mPa.s')
# ax.scatter(t_amp4, A4, label='100 mPa.s')
# ax.scatter(t_amp7, A7, label='200 mPa.s')
# handles,labels = ax.get_legend_handles_labels()
# handles = [handles[0], handles[1], handles[2]]
# labels = [labels[0], labels[1], labels[2]]
# ax.legend(handles,labels, loc=1, fontsize=12.5)
# plt.tick_params(axis='both', which='major', labelsize=12.5)
# plt.tick_params(axis='both', which='minor', labelsize=12.5)
#
# plt.subplot(gs[0, 4:7])
# ax = plt.gca()
# plt.text(0.2, 6000, '10 $\mu m$', fontsize=15)
# ax.set_yscale('log')
# ax.set_xscale('log')
# plt.xlim(0.01,10)
# plt.ylim(10,10000)
# plt.xlabel(r'$t$  $[s]$', fontsize=15)
# plt.ylabel(r'$A_{\delta}$ $[nm]$', fontsize=15)
# ax.scatter(t_amp2, A2, label='050 mPa.s')
# ax.scatter(t_amp5, A5, label='100 mPa.s')
# ax.scatter(t_amp8, A8, label='200 mPa.s')
# handles,labels = ax.get_legend_handles_labels()
# handles = [handles[0], handles[1], handles[2]]
# labels = [labels[0], labels[1], labels[2]]
# ax.legend(handles,labels, loc=1, fontsize=12.5)
# plt.tick_params(axis='both', which='major', labelsize=12.5)
# plt.tick_params(axis='both', which='minor', labelsize=12.5)
#
# plt.subplot(gs[1, 2:5])
# ax = plt.gca()
# plt.text(0.2, 6000, '15 $\mu m$', fontsize=15)
# ax.set_yscale('log')
# ax.set_xscale('log')
# plt.xlim(0.01,10)
# plt.ylim(10,10000)
# plt.xlabel(r'$t$  $[s]$', fontsize=15)
# plt.ylabel(r'$A_{\delta}$ $[nm]$', fontsize=15)
# ax.scatter(t_amp3, A3, label='050 mPa.s')
# ax.scatter(t_amp6, A6, label='100 mPa.s')
# ax.scatter(t_amp9, A9, label='200 mPa.s')
# handles,labels = ax.get_legend_handles_labels()
# handles = [handles[0], handles[1], handles[2]]
# labels = [labels[0], labels[1], labels[2]]
# ax.legend(handles,labels, loc=1, fontsize=12.5)
# plt.tick_params(axis='both', which='major', labelsize=12.5)
# plt.tick_params(axis='both', which='minor', labelsize=12.5)
#
# plt.show()

################################################################################
# #dimensionalized wavelength variation with time
# fig, ax = plt.subplots(2,2)
#
# plt.subplot(2,2,1)
# ax = plt.gca()
# plt.title('050 cSt')
# ax.set_yscale('log')
# ax.set_xscale('log')
# plt.xlim(0.01,10)
# plt.ylim(10,10000)
# plt.xlabel(r'$t$ $[s]$')
# plt.ylabel(r'$L_{\lambda}$ $[nm]$')
# ax.scatter(t_amp1, L1, label='05 $\mu m$')
# ax.scatter(t_amp2, L2, label='10 $\mu m$')
# ax.scatter(t_amp3, L3, label='15 $\mu m$')
# handles,labels = ax.get_legend_handles_labels()
# handles = [handles[0], handles[1], handles[2]]
# labels = [labels[0], labels[1], labels[2]]
# ax.legend(handles,labels)
# #ax.grid(True,which='both',ls='-')
# # plt.savefig('amplitude_decay.png')
#
# plt.subplot(2,2,2)
# ax = plt.gca()
# plt.title('100 cSt')
# ax.set_yscale('log')
# ax.set_xscale('log')
# plt.xlim(0.01,10)
# plt.ylim(10,10000)
# plt.xlabel(r'$t$ $[s]$')
# plt.ylabel(r'$L_{\lambda}$ $[nm]$')
# ax.scatter(t_amp4, L4, label='05 $\mu m$')
# ax.scatter(t_amp5, L5, label='10 $\mu m$')
# ax.scatter(t_amp6, L6, label='15 $\mu m$')
# handles,labels = ax.get_legend_handles_labels()
# handles = [handles[0], handles[1], handles[2]]
# labels = [labels[0], labels[1], labels[2]]
# ax.legend(handles,labels)
# #ax.grid(True,which='both',ls='-')
# # plt.savefig('amplitude_decay.png')
#
# plt.subplot(2,2,3)
# ax = plt.gca()
# plt.title('200 cSt')
# ax.set_yscale('log')
# ax.set_xscale('log')
# plt.xlim(0.01,10)
# plt.ylim(10,10000)
# plt.xlabel(r'$t$ $[s]$')
# plt.ylabel(r'$L_{\lambda}$ $[nm]$')
# ax.scatter(t_amp7, L7, label='05 $\mu m$')
# ax.scatter(t_amp8, L8, label='10 $\mu m$')
# ax.scatter(t_amp9, L9, label='15 $\mu m$')
# handles,labels = ax.get_legend_handles_labels()
# handles = [handles[0], handles[1], handles[2]]
# labels = [labels[0], labels[1], labels[2]]
# ax.legend(handles,labels)
# #ax.grid(True,which='both',ls='-')
# # plt.savefig('amplitude_decay.png')
#
# plt.subplot(2,2,4)
# ax = plt.gca()
# plt.title('350 cSt')
# ax.set_yscale('log')
# ax.set_xscale('log')
# plt.xlim(0.01,10)
# plt.ylim(10,10000)
# plt.xlabel(r'$t$ $[s]$')
# plt.ylabel(r'$L_{\lambda}$ $[nm]$')
# # ax.scatter(t_amp7, L7, label='05 $\mu m$')
# # ax.scatter(t_amp8, L8, label='10 $\mu m$')
# # ax.scatter(t_amp9, L9, label='15 $\mu m$')
# # handles,labels = ax.get_legend_handles_labels()
# # handles = [handles[0], handles[1], handles[2]]
# # labels = [labels[0], labels[1], labels[2]]
# # ax.legend(handles,labels)
# #ax.grid(True,which='both',ls='-')
# # plt.savefig('amplitude_decay.png')
#
# plt.show()

################################################################################
