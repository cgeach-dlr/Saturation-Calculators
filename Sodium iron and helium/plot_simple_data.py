# -*- coding: utf-8 -*-

import sys
sys.path.append("")
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

font = {'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

#Designate an output location for figure data and plots
outpath = os.path.join(os.path.dirname(os.getcwd()), 'Output')
 
Data_simple_Delta_nu_fname = os.path.join(outpath, 'Figure Data',
                                     'Simple_Delta_nu_L_data.txt')
Data_simple_Delta_nu = np.loadtxt(Data_simple_Delta_nu_fname,
           delimiter=',').T
Data_simple_spectrum_fname = os.path.join(outpath, 'Figure Data',
                                     'Simple_nu_L_data.txt')
Data_simple_spectrum = np.loadtxt(Data_simple_spectrum_fname,
           delimiter=',').T
    
Delta_nu_Ls = Data_simple_Delta_nu[0]
sats_vdG_Delta_nu_Ls = Data_simple_Delta_nu[1]
sats_Megie_Delta_nu_Ls = Data_simple_Delta_nu[2]
sats_Megie_Delta_nu_Ls2 = Data_simple_Delta_nu[3]

nu_Ls = Data_simple_spectrum[0]
sats_vdG_nu_Ls = Data_simple_spectrum[1]
sats_Megie_nu_Ls = Data_simple_spectrum[2]
sats_Megie_nu_Ls2 = Data_simple_spectrum[3]
  
fig,ax = plt.subplots(1,2, figsize=(16,8))

ax[0].plot(Delta_nu_Ls, sats_vdG_Delta_nu_Ls,
           label='von der Gathen approach')
ax[0].plot(Delta_nu_Ls, sats_Megie_Delta_nu_Ls,
           label=r'Megie approach ($\sigma_{\mathrm{e}}$)')
ax[0].plot(Delta_nu_Ls, sats_Megie_Delta_nu_Ls2,
           label=r'Megie approach ($\sigma_{\mathrm{t}}$)')

ax[0].text(.09,.9, '(a)', transform=plt.gcf().transFigure)
ax[0].set_ylabel('Saturation percent')
ax[0].set_xlabel('Laser linewidth (MHz)')
ax[0].legend()
ax[0].grid(True)

ax[1].text(.51,.9, '(b)', transform=plt.gcf().transFigure)
ax[1].plot(nu_Ls, sats_vdG_nu_Ls)
ax[1].plot(nu_Ls, sats_Megie_nu_Ls)
ax[1].plot(nu_Ls, sats_Megie_nu_Ls2)

ax[1].set_xlabel('Laser frequency offset (GHz)')
ax[1].grid(True)

plt.savefig(os.path.join(outpath, 'Simple.pdf'), dpi=300)
