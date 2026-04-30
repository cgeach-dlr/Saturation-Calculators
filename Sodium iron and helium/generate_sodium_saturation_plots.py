# -*- coding: utf-8 -*-

import sys
sys.path.append("")
import sodium_saturation_calculator_library as na_lib
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

#Designate an output location for figure data and plots
outpath = os.path.join(os.path.dirname(os.getcwd()), 'Output')

#Calculates the spectrum of the degree of saturation, following the Megie and
# VDG approaches.
   
nu_Ls = (np.arange(68) - 33)*1e8 #Hz
sats_vdG_corr = np.zeros(len(nu_Ls))
sats_vdG_uncorr = np.zeros(len(nu_Ls))
sats_Megie = np.zeros(len(nu_Ls))

E_pulse = 20 #mJ
N_L = na_lib.N_L_from_pulse_energy(E_pulse)
t_L = 6.7 #ns
T_atm = 0.7
Temp_Na = 200 #K
z = 90000 #km
alpha_L = 450e-6 #radians
alpha_T = 600e-6 #radians
lineshape = 'gauss'

nt = 100
delta_t = 1 #ns
delta_r = 150e-6 #radians

Delta_nu_L = 130e6 #Hz
Doppler_spectrum = na_lib.get_total_scattering_cross_section_spectrum(Temp_Na)

for i in range(len(nu_Ls)):
    nu_L = nu_Ls[i]    
    sats_vdG_corr[i] = na_lib.get_saturation_beam(nu_L, Delta_nu_L, N_L, z,
                  T_atm, alpha_L, alpha_T, t_L, nt, delta_t, delta_r, Temp_Na,
                  lineshape, ratio_beam=True)
    sats_vdG_uncorr[i] = na_lib.get_saturation_beam_VDG(nu_L, Delta_nu_L, N_L,
                  z, T_atm, alpha_L, alpha_T, t_L, nt, delta_t, delta_r,
                  Temp_Na, lineshape, ratio_beam=True)
    g_L = na_lib.get_laser_pulseshape(nu_L, Delta_nu_L, lineshape)
    sigma_eff = np.sum(g_L * Doppler_spectrum) / np.sum(g_L)
    sats_Megie[i] = na_lib.get_saturation_megie(z, alpha_L, t_L, sigma_eff, N_L,
                  T_atm)
    
nt = 400
delta_t = 0.25 #ns
T_atm = 1
Omega = np.pi / 4 * alpha_L**2   

Es = 1.288 * 10**np.arange(0, 2.1, 0.2) #mJ (The factor 1.288 scales the 
                                         # maxiumum pulse energy such that the 
                                         # energy density is 0.1 mJ / m^2)
Es = np.hstack((1e-3, Es))

u_0s = Es / z**2 / Omega

dens_err = np.zeros(len(Es)) #Arbitrary units
T_err = np.zeros(len(Es)) #K
w_err = np.zeros(len(Es)) #m/s

nu_Ls_errs = np.array([-1.27e9, -6.4e8, -0.1e8]) #Hz

for i in range(len(Es)):
    N_L = na_lib.N_L_from_pulse_energy(Es[i])
    Res = na_lib.get_wind_and_temp_errors(Temp_Na, nu_Ls_errs, Delta_nu_L, N_L,
                                          z, T_atm, alpha_L, alpha_T, t_L, nt,
                                          delta_t, delta_r, lineshape)
      
    dens_err[i] = (Res[0][0][0] - Res[1][0][0])/Res[1][0][0]
    T_err[i] = Res[0][0][1] - Res[1][0][1]
    w_err[i] = (Res[0][0][2] - Res[1][0][2])
    
fig,ax = plt.subplots(1,2, figsize=(16,8))

ax[0].plot(nu_Ls*1e-9, 100*sats_vdG_corr,
           label='VDG approach\n (corrected line strengths)')
ax[0].plot(nu_Ls*1e-9, 100*sats_vdG_uncorr,
           label='VDG approach\n (uncorrected line strengths)')
ax[0].plot(nu_Ls*1e-9, 100*sats_Megie,
           label=r'Megie approach ($\sigma_{\mathrm{t}}$)')

ax[0].text(.06,.92, '(a)', transform=plt.gcf().transFigure)
ax[0].set_ylabel('Saturation percent')
ax[0].set_xlabel('Laser frequency offset (GHz)')
ax[0].set_xlim(-3.5,3.5)
ax[0].set_ylim(-0.5,11.9)
ax[0].legend()
ax[0].grid(True)

ax[1].text(.51,.93, '(b)', transform=plt.gcf().transFigure)
ax[1].plot(u_0s, dens_err, label='Density bias')
ax[1].plot(u_0s, T_err, label='Temperature bias')
ax[1].plot(u_0s, w_err, label='Wind bias')
ax[1].axvline(E_pulse / z**2 / Omega * T_atm, ls='--', c='k')

ax[1].set_ylim(-1.5,2.5)

ax[1].set_xlabel(r'Laser energy density (mJ/m$^2$)')
ax[1].set_ylabel('Fractional density bias \n Absolute temperature bias ' + 
                 '(K)\n Absolute wind bias (m/s)')
ax[1].grid(True)
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].legend(loc='upper left', framealpha=1)

fig.tight_layout()

plt.savefig(os.path.join(outpath, 'Na.pdf'), dpi=300)


Na_spectrum_data = np.vstack((nu_Ls*1e-9, 100*sats_vdG_corr,
                              100*sats_vdG_uncorr, 100*sats_Megie))
np.savetxt(os.path.join(outpath, 'Na_saturation.txt'), Na_spectrum_data.T,
           delimiter=',')

Na_temp_and_wind_biases_data = np.vstack((u_0s, dens_err, T_err, w_err))
np.savetxt(os.path.join(outpath, 'Na_temp_and_wind_biases.txt'),
           Na_temp_and_wind_biases_data.T, delimiter=',')
