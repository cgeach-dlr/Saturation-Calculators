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

Data_Fig_a = np.loadtxt(os.path.join(outpath, 'Figure Data',
                                     'He_saturation_SI.txt'),
           delimiter=',').T
Data_Fig_b = np.loadtxt(os.path.join(outpath, 'Figure Data',
                                     'He_temp_and_wind_biases_SI.txt'),
           delimiter=',').T
          
nu_Ls = Data_Fig_a[0]
sats_vdG_nu_Ls = Data_Fig_a[1]
sats_Megie_nu_Ls = Data_Fig_a[2]

u_0s = Data_Fig_b[0]
dens_err = Data_Fig_b[1]
T_err = Data_Fig_b[2]
w_err = Data_Fig_b[3]

E_pulse = 4.7 #mJ
z = 500000 #m
alpha_L = 30e-6 #rad
Omega = np.pi / 4 * np.sin(alpha_L)**2
T_atm = 0.9

fig,ax = plt.subplots(1,2, figsize=(16,8))

ax[0].plot(nu_Ls, sats_vdG_nu_Ls, label='von der Gathen approach')
ax[0].plot(nu_Ls, sats_Megie_nu_Ls,
           label=r'Megie approach ($\sigma_{\mathrm{t}}$)')

ax[0].text(.06,.92, '(a)', transform=plt.gcf().transFigure)
ax[0].set_ylabel('Saturation percent')
ax[0].set_xlabel('Laser frequency offset (GHz)')
ax[0].set_ylim(-1,19)
ax[0].legend()
ax[0].grid(True)

ax[1].text(.51,.93, '(b)', transform=plt.gcf().transFigure)
ax[1].plot(u_0s, dens_err, label = 'Density bias')
ax[1].plot(u_0s, T_err, label = 'Temperature bias')
ax[1].plot(u_0s, w_err, label = 'Wind bias')
ax[1].axvline(E_pulse / z**2 / Omega * T_atm, ls='--', c='k')

ax[1].set_xlabel(r'Laser energy density (mJ/m$^2$)')
ax[1].set_ylabel('Percent density bias \n Absolute temperature bias (K)\n' +
                 'Absolute wind bias (m/s)')
ax[1].legend()
ax[1].grid(True)
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
fig.tight_layout()

plt.savefig(os.path.join(outpath, 'He_SI.pdf'), dpi=300)
