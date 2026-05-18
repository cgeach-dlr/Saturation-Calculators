# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 08:22:33 2026

@author: geac_ch
"""

import sys
sys.path.append("")
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)
      
outpath = os.path.join(os.path.dirname(os.getcwd()), 'Output')

Data_Na_saturation_fname = os.path.join(outpath, 'Figure Data',
                                     'Na_saturation_SI.txt')
Data_Na_saturation = np.loadtxt(Data_He_saturation_fname,
           delimiter=',').T
Data_Na_biases_fname = os.path.join(outpath, 'Figure Data',
                                     'Na_temp_and_wind_biases_SI.txt')
Data_Na_biases = np.loadtxt(Data_He_biases_fname,
           delimiter=',').T
          
nu_Ls = Data_Na_saturation[0]
sats_vdG_corr = Data_Na_saturation[1]
sats_vdG_uncorr = Data_Na_saturation[2]
sats_Megie = Data_Na_saturation[3]

u_0s = Data_Na_biases[0]
dens_err = Data_Na_biases[1]
T_err = Data_Na_biases[2]
w_err = Data_Na_biases[3]

E_pulse = 20 #mJ
z = 90000 #m
alpha_L = 450e-6 #rad
Omega = np.pi / 4 * np.sin(alpha_L)**2
T_atm = 0.7 

fig,ax = plt.subplots(1,2, figsize=(16,8))

ax[0].plot(nu_Ls, sats_vdG_corr,
           label='VDG approach\n (corrected line strengths)')
ax[0].plot(nu_Ls, sats_vdG_uncorr,
           label='VDG approach\n (uncorrected line strengths)')
ax[0].plot(nu_Ls, sats_Megie, label=r'Megie approach ($\sigma_{\mathrm{t}}$)')

ax[0].text(.06,.92, '(a)', transform=plt.gcf().transFigure)
ax[0].set_ylabel('Saturation percent')
ax[0].set_xlabel('Laser frequency offset (GHz)')
ax[0].set_xlim(-3.5,3.5)
ax[0].set_ylim(-0.5,8.9)
ax[0].legend()
ax[0].grid(True)

ax[1].text(.51,.93, '(b)', transform=plt.gcf().transFigure)
ax[1].plot(u_0s, dens_err, label='Density bias')
ax[1].plot(u_0s, T_err, label='Temperature bias')
ax[1].plot(u_0s, w_err, label='Wind bias')
ax[1].axvline(E_pulse / z**2 / Omega * T_atm, ls='--', c='k')

ax[1].set_ylim(-0.5,12.5)

ax[1].set_xlabel(r'Laser energy density (mJ/m$^2$)')
ax[1].set_ylabel('Fractional density bias \n Absolute temperature bias ' + 
                 '(K)\n Absolute wind bias (m/s)')
ax[1].grid(True)
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].legend(loc='upper left', framealpha=1)

fig.tight_layout()

plt.savefig(os.path.join(outpath, 'Na_SI.pdf'), dpi=300)

fig,ax = plt.subplots(1,2, figsize=(16,8))

ax[0].plot(nu_Ls, sats_vdG_corr,
           label='VDG approach\n (corrected line strengths)')
ax[0].plot(nu_Ls, sats_vdG_uncorr,
           label='VDG approach\n (uncorrected line strengths)')
ax[0].plot(nu_Ls, sats_Megie, label=r'Megie approach ($\sigma_{\mathrm{t}}$)')

ax[0].text(.06,.92, '(a)', transform=plt.gcf().transFigure)
ax[0].set_ylabel('Saturation percent')
ax[0].set_xlabel('Laser frequency offset (GHz)')
ax[0].set_xlim(-3.5,3.5)
ax[0].set_ylim(-0.5,8.9)
ax[0].legend()
ax[0].grid(True)

ax[1].text(.51,.93, '(b)', transform=plt.gcf().transFigure)
ax[1].plot(u_0s, dens_err, label='Density bias')
ax[1].plot(u_0s, T_err, label='Temperature bias')
ax[1].plot(u_0s, w_err, label='Wind bias')
ax[1].axvline(E_pulse / z**2 / Omega * T_atm, ls='--', c='k')

ax[1].set_ylim(-0.5,12.5)

ax[1].set_xlabel(r'Laser energy density (mJ/m$^2$)')
ax[1].set_ylabel('Fractional density bias \n Absolute temperature bias ' + 
                 '(K)\n Absolute wind bias (m/s)')
ax[1].grid(True)
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].legend(loc='upper left', framealpha=1)

fig.tight_layout()

plt.savefig(os.path.join(outpath, 'Na_SI.png'), dpi=300)
