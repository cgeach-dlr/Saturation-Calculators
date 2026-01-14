# -*- coding: utf-8 -*-

import sys
sys.path.append("")
import helium_saturation_calculator_library as he_lib
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

#Designate an output location for figure data and plots
outpath = os.path.join(os.path.dirname(os.getcwd()), 'Output')

#Calculates the spectrum of the degree of saturation.

nu_Ls = (np.arange(59) - 29)*4e8 #Hz
sats_vdG_nu_Ls = np.zeros(len(nu_Ls))
sats_Megie_nu_Ls = np.zeros(len(nu_Ls))

E_pulse = 4.7 #mJ
t_L = 200 #ns
T_atm = 0.9
z = 500000 #km
alpha_L = 30e-6 #radians
alpha_T = 61e-6 #radians
delta_r = 31e-6 #radians

nt = 200
delta_t = 5 #ns

N_L = he_lib.N_L_from_pulse_energy(E_pulse)
Delta_nu_L = 20e6 #Hz
lineshape = 'gauss'
Temp_He = 1000 #K
Doppler_spectrum = he_lib.get_total_scattering_cross_section_spectrum(Temp_He)

for i in range(len(nu_Ls)):
    print(i)
    nu_L = nu_Ls[i]    
    sats_vdG_nu_Ls[i] = he_lib.get_saturation_beam(nu_L, Delta_nu_L, N_L, z,
                          T_atm, alpha_L, alpha_T, t_L, nt, delta_t, delta_r,
                          Temp_He, lineshape, ratio_beam=True)    
    g_L = he_lib.get_laser_pulseshape(nu_L, Delta_nu_L, lineshape)
    sigma_eff = np.sum(g_L * Doppler_spectrum) / np.sum(g_L)
    sats_Megie_nu_Ls[i] = he_lib.get_saturation_megie(z, alpha_L, t_L,
                          sigma_eff, N_L, T_atm)


#Calculates the saturation-induced density, temperature, and wind biases as a
# function of laser pulse energy

delta_ts = np.array([10, 10, 8, 5, 3.2, 2, 1.25, 0.8, 0.5, 0.32, 0.2, 0.125])
nts = 1000 / delta_ts

nu_Ls_errs = np.array([-3.58e9, 0, 2.04e9, 3.32e9])

Es = 10**np.arange(0, 2.1, 0.2) #mJ
Es = np.hstack((1e-3, Es))

dens_err_200 = np.zeros(len(Es)) #Arbitrary units
T_err_200 = np.zeros(len(Es)) #K
w_err_200 = np.zeros(len(Es)) #m/s

for i in range(len(Es)):
    N_L = he_lib.N_L_from_pulse_energy(Es[i])
    Res = he_lib.get_wind_and_temp_errors(Temp_He, nu_Ls_errs, Delta_nu_L,
                                          N_L, z, T_atm, alpha_L, alpha_T,
                                          t_L, nts[i], delta_ts[i], delta_r,
                                          lineshape)
      
    dens_err_200[i] = (Res[0][0][0] - Res[1][0][0])/Res[1][0][0]
    T_err_200[i] = Res[0][0][1] - Res[1][0][1]
    w_err_200[i] = Res[0][0][2] - Res[1][0][2]  

fig,ax = plt.subplots(1,2, figsize=(16,8))

ax[0].plot(nu_Ls*1e-9, 100*sats_vdG_nu_Ls, label='von der Gathen approach')
ax[0].plot(nu_Ls*1e-9, 100*sats_Megie_nu_Ls,
           label=r'Megie approach ($\sigma_{\mathrm{t}}$)')

ax[0].text(.06,.92, '(a)', transform=plt.gcf().transFigure)
ax[0].set_ylabel('Saturation percent')
ax[0].set_xlabel('Central laser frequency (GHz)')
ax[0].set_ylim(-1,34)
ax[0].legend()
ax[0].grid(True)

ax[1].text(.51,.93, '(b)', transform=plt.gcf().transFigure)
ax[1].plot(Es, 100*dens_err_200, label = 'Density error')
ax[1].plot(Es, T_err_200, label = 'Temperature error')
ax[1].plot(Es, w_err_200, label = 'Wind error')

ax[1].set_xlabel('Laser pulse energy (mJ)')
ax[1].set_ylabel('Percent density error \n Absolute temperature error (K)\n' +
                 'Absolute wind error (m/s)')
ax[1].legend()
ax[1].grid(True)
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
fig.tight_layout()

plt.savefig(os.path.join(outpath, 'He.pdf'), dpi=300)

He_spectrum_data = np.vstack((nu_Ls*1e-9, 100*sats_vdG_nu_Ls, 
                              100*sats_Megie_nu_Ls))
np.savetxt(os.path.join(outpath, 'He_saturation.txt'),
           He_spectrum_data.T, delimiter=',')

He_temp_and_wind_biases_data = np.vstack((Es, 100*dens_err_200, T_err_200,
                                          w_err_200))
np.savetxt(os.path.join(outpath, 'He_temp_and_wind_biases.txt'),
           He_temp_and_wind_biases_data.T, delimiter=',')





