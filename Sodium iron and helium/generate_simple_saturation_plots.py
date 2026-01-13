# -*- coding: utf-8 -*-

import sys
sys.path.append("")
import simple_saturation_calculator_library as sat_lib
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

font = {'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

c_light = 2.99792458 * 10**8 #m/s
lamb0 =  589.158 * 10**-9 #m, in vacuum
nu0 = c_light / lamb0
  
#Designate an output location for figure data and plots
outpath_data = ''
outpath_figs = ''
  
#Calculates the degree of saturation for a range of laser linewidths, following 
# the Megie and VDG approaches.
    
Delta_nu_Ls = (np.arange(100)+1)*2e7 #Hz 

sats_vdG_Delta_nu_Ls = np.zeros(len(Delta_nu_Ls))
sats_Megie_Delta_nu_Ls = np.zeros(len(Delta_nu_Ls))
sats_Megie_Delta_nu_Ls2 = np.zeros(len(Delta_nu_Ls))

E_pulse = 20 #mJ
N_L = sat_lib.N_L_from_pulse_energy(E_pulse)
t_L = 10 #ns
T_atm = 0.7 
Temp_layer = 200 #K
z = 90000 #km
alpha_L = 0.001 #rad

nt = 100
delta_t = 1

nu_L = 0
absorption_spectrum =  sat_lib.get_natural_absorption_line()
Doppler_spectrum = sat_lib.get_doppler_broadened_spectrum(Temp_layer)
temp_spectrum = sat_lib.get_temperature_spectrum(Temp_layer)

for i in range(len(Delta_nu_Ls)):
    Delta_nu_L = Delta_nu_Ls[i]
    sats_vdG_Delta_nu_Ls[i] = sat_lib.get_saturation(nu_L, Delta_nu_L, N_L, z,
                                  alpha_L, T_atm, t_L, nt, delta_t, Temp_layer,
                                  ratio=True)
    g_L = sat_lib.g_L_gauss(nu_L, Delta_nu_L)
    sigma_eff = np.sum(g_L * absorption_spectrum) / np.sum(g_L)
    sigma_eff2 = np.sum(g_L * Doppler_spectrum) / np.sum(g_L)
    
    sats_Megie_Delta_nu_Ls[i] = sat_lib.get_saturation_megie(z, alpha_L, t_L,
                                  sigma_eff, N_L, T_atm)
    sats_Megie_Delta_nu_Ls2[i] = sat_lib.get_saturation_megie(z, alpha_L, t_L, 
                                  sigma_eff2, N_L, T_atm)



#Calculates the spectrum of the degree of saturation, following the Megie and
#  VDG approaches.

nu_Ls = (np.arange(301) - 150)*2e7
sats_vdG_nu_Ls = np.zeros(len(nu_Ls))

Delta_nu_L = 1.5e8

for i in range(len(nu_Ls)):
    nu_L = nu_Ls[i]    
    sats_vdG_nu_Ls[i] = sat_lib.get_saturation(nu_L, Delta_nu_L, N_L, z,  alpha_L,
                  T_atm, t_L, nt, delta_t, Temp_layer, ratio=True)
    
#nu_Ls_Megie = (np.arange(601) - 300)*1e7
nu_Ls_Megie = (np.arange(301) - 150)*2e7
sats_Megie_nu_Ls = np.zeros(len(nu_Ls_Megie))
sats_Megie_nu_Ls2 = np.zeros(len(nu_Ls_Megie))

for i in range(len(nu_Ls_Megie)):
    nu_L = nu_Ls_Megie[i]
    g_L = sat_lib.g_L_gauss(nu_L, Delta_nu_L)
    sigma_eff = np.sum(g_L * absorption_spectrum) / np.sum(g_L)
    sigma_eff2 = np.sum(g_L * Doppler_spectrum) / np.sum(g_L)
    
    sats_Megie_nu_Ls[i] = sat_lib.get_saturation_megie(z, alpha_L, t_L,
                                                       sigma_eff, N_L, T_atm)
    sats_Megie_nu_Ls2[i] = sat_lib.get_saturation_megie(z, alpha_L, t_L,
                                                       sigma_eff2, N_L, T_atm)
  
fig,ax = plt.subplots(1,2, figsize=(15,7))

ax[0].plot(Delta_nu_Ls*1e-6, 100*sats_vdG_Delta_nu_Ls,
           label='von der Gathen approach')
ax[0].plot(Delta_nu_Ls*1e-6, 100*sats_Megie_Delta_nu_Ls,
           label=r'Megie approach ($\sigma_{\mathrm{e}}$)')
ax[0].plot(Delta_nu_Ls*1e-6, 100*sats_Megie_Delta_nu_Ls2,
           label=r'Megie approach ($\sigma_{\mathrm{t}}$)')

ax[0].text(.09,.9, '(a)', transform=plt.gcf().transFigure)
ax[0].set_ylabel('Saturation percent')
ax[0].set_xlabel('Laser line width (MHz)')
ax[0].legend()
ax[0].grid(True)

ax[1].text(.51,.9, '(b)', transform=plt.gcf().transFigure)
ax[1].plot(nu_Ls*1e-6, 100*sats_vdG_nu_Ls)
ax[1].plot(nu_Ls_Megie*1e-6, 100*sats_Megie_nu_Ls)
ax[1].plot(nu_Ls_Megie*1e-6, 100*sats_Megie_nu_Ls2)

ax[1].set_xlabel('Central laser frequency(MHz)')
ax[1].grid(True)

plt.savefig(os.path.join(outpath_figs, 'Simple.pdf'), dpi=300)

Delta_nu_L_data = np.vstack((Delta_nu_Ls*1e-6, 100*sats_vdG_Delta_nu_Ls, 
                    100*sats_Megie_Delta_nu_Ls, 100*sats_Megie_Delta_nu_Ls2))
np.savetxt(os.path.join(outpath_data, 'Simple_Delta_nu_L_data.txt'),
           Delta_nu_L_data.T, delimiter=',')

nu_L_data = np.vstack((nu_Ls*1e-6, 100*sats_vdG_nu_Ls, 100*sats_Megie_nu_Ls,
                       100*sats_Megie_nu_Ls2))
np.savetxt(os.path.join(outpath_data, 'Simple_nu_L_data.txt'), nu_L_data.T,
           delimiter=',')
