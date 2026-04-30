# -*- coding: utf-8 -*-

import sys
sys.path.append("")
import calcium_saturation_calculator_library as ca_lib
import calcium_plus_saturation_calculator_library as ca_plus_lib
import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib

font = {'weight' : 'normal',
        'size'   : 24}

matplotlib.rc('font', **font)

#Designate an output location for figure data and plots
outpath = os.path.join(os.path.dirname(os.getcwd()), 'Output')

#Calculates the spectra of the degree of saturation for Figures 1 and 3

lambda_Ls = (np.arange(25) - 11)*0.15e-12
ca_nu_Ls = lambda_Ls / ca_lib.lamb0 * ca_lib.nu0
ca_plus_nu_Ls = lambda_Ls / ca_plus_lib.lamb0 * ca_plus_lib.nu0

ca_sats1 = np.zeros(len(lambda_Ls))
ca_sats2 = np.zeros(len(lambda_Ls))
ca_sats3 = np.zeros(len(lambda_Ls))
ca_plus_sats1 = np.zeros(len(lambda_Ls))
ca_plus_sats2 = np.zeros(len(lambda_Ls))
ca_plus_sats3 = np.zeros(len(lambda_Ls))

ca_E_pulse = 5 #mJ
ca_N_L = ca_lib.N_L_from_pulse_energy(ca_E_pulse)
ca_t_L = 13.2 #ns
ca_Delta_nu_L = 25.3e6 #Hz

ca_plus_E_pulse = 4 #mJ
ca_plus_N_L = ca_plus_lib.N_L_from_pulse_energy(ca_plus_E_pulse)
ca_plus_t_L = 11.9 #ns
ca_plus_Delta_nu_L = 35.2e6 #Hz

T_atm = 0.7
Temp_Ca = 200 #K
z = 90000 #km
alpha_L = 200e-6 #radians
alpha_L2 = 300e-6 #radians
alpha_T = 800e-6 #radians
lineshape = 'lorentzian'

nt = 500
delta_t = 0.2 #ns
delta_r = 100e-6 #radians

Doppler_spectrum = ca_lib.get_total_scattering_cross_section_spectrum(Temp_Ca)

for i in range(len(lambda_Ls)):
    ca_sats1[i] = ca_lib.get_saturation_beam(ca_nu_Ls[i], ca_Delta_nu_L, ca_N_L,
                z, T_atm, alpha_L, alpha_T, ca_t_L, nt, delta_t, delta_r,
                Temp_Ca, lineshape)
    ca_sats2[i] = ca_lib.get_saturation_beam(ca_nu_Ls[i], ca_Delta_nu_L/10,
                ca_N_L, z, T_atm, alpha_L, alpha_T, 10*ca_t_L, nt, 10*delta_t,
                delta_r, Temp_Ca, lineshape)
    ca_sats3[i] = ca_lib.get_saturation_beam(ca_nu_Ls[i], ca_Delta_nu_L/10,
                ca_N_L, z, T_atm, alpha_L2, alpha_T, 10*ca_t_L, nt, 10*delta_t,
                delta_r, Temp_Ca, lineshape)    
    ca_plus_sats1[i] = ca_plus_lib.get_saturation_beam(ca_plus_nu_Ls[i], 
                ca_plus_Delta_nu_L, ca_plus_N_L, z, T_atm, alpha_L, alpha_T,
                ca_plus_t_L, nt, delta_t, delta_r, Temp_Ca, lineshape)
    ca_plus_sats2[i] = ca_plus_lib.get_saturation_beam(ca_plus_nu_Ls[i],
                ca_plus_Delta_nu_L/10, ca_plus_N_L, z, T_atm, alpha_L, alpha_T,
                10*ca_plus_t_L, nt, 10*delta_t, delta_r, Temp_Ca, lineshape)
    ca_plus_sats3[i] = ca_plus_lib.get_saturation_beam(ca_plus_nu_Ls[i],
                ca_plus_Delta_nu_L/10, ca_plus_N_L, z, T_atm, alpha_L2, alpha_T,
                10*ca_plus_t_L, nt, 10*delta_t, delta_r, Temp_Ca, lineshape)

Spectrum_data = np.vstack((lambda_Ls*1e12, 100*ca_sats1, 100*ca_sats2,
                           100*ca_sats3, 100*ca_plus_sats1, 100*ca_plus_sats2,
                           100*ca_plus_sats3))

np.savetxt(os.path.join(outpath, 'Ca_and_ca_plus_spectrum.txt'),
           Spectrum_data.T, delimiter=',')

# Generate plot for Figure 1
plt.figure(figsize=(12,8))

plt.plot(lambda_Ls*1e12, 100*ca_sats1, label='Ca')
plt.plot(lambda_Ls*1e12, 100*ca_plus_sats1, label='Ca+')
plt.legend()
plt.ylabel('Saturation percent')
plt.xlabel('Laser wavelength offset (pm)')

plt.grid(True)

plt.savefig(os.path.join(outpath, 'Ca_and_ca_plus1.jpg'), dpi=300)

# Generate plot for Figure 3
plt.figure(figsize=(12,8))

plt.plot(lambda_Ls*1e12, 100*ca_sats2, c='tab:blue', label='Ca')
plt.plot(lambda_Ls*1e12, 100*ca_sats3, c='tab:blue', linestyle=':')
plt.plot(lambda_Ls*1e12, 100*ca_plus_sats2, c='tab:orange',
         label='Ca+')
plt.plot(lambda_Ls*1e12, 100*ca_plus_sats3, c='tab:orange',
         linestyle=':')
plt.legend()
plt.ylabel('Saturation percent')
plt.xlabel('Laser wavelength offset (pm)')

plt.grid(True)

plt.savefig(os.path.join(outpath, 'Ca_and_ca_plus3.jpg'), dpi=300)

# Calculates the saturation-induces temperature biases for Figures 2 and 4
Es = 10**np.arange(0, 2.1, 0.2) #mJ
Es = np.hstack((1e-3, Es))
nts = np.array([500, 500, 500, 500, 500, 500, 500, 800, 1200, 1800, 2800,
                4400])

ca_T_err1 = np.zeros(len(Es)) #K
ca_T_err2 = np.zeros(len(Es)) #K
ca_T_err3 = np.zeros(len(Es)) #K
ca_plus_T_err1 = np.zeros(len(Es)) #K
ca_plus_T_err2 = np.zeros(len(Es)) #K
ca_plus_T_err3 = np.zeros(len(Es)) #K

lambda_Ls_errs = (np.arange(6)-2) * 0.24e-12
ca_nu_Ls_errs = lambda_Ls_errs / ca_lib.lamb0 * ca_lib.nu0
ca_plus_nu_Ls_errs = lambda_Ls_errs / ca_plus_lib.lamb0 * ca_plus_lib.nu0

for i in range(len(Es)):
    ca_N_L = ca_lib.N_L_from_pulse_energy(Es[i])
    ca_Res1 = ca_lib.get_wind_and_temp_errors(Temp_Ca, ca_nu_Ls_errs,
                                             ca_Delta_nu_L, ca_N_L, z, T_atm,
                                             alpha_L, alpha_T, ca_t_L, nts[i],
                                             100/nts[i], delta_r, lineshape)
    ca_Res2 = ca_lib.get_wind_and_temp_errors(Temp_Ca, ca_nu_Ls_errs,
                                              ca_Delta_nu_L/10, ca_N_L, z,
                                              T_atm, alpha_L, alpha_T,
                                              10*ca_t_L, nts[i], 1000/nts[i],
                                              delta_r, lineshape)
    ca_Res3 = ca_lib.get_wind_and_temp_errors(Temp_Ca, ca_nu_Ls_errs,
                                              ca_Delta_nu_L/10, ca_N_L, z,
                                              T_atm, alpha_L2, alpha_T,
                                              10*ca_t_L, nts[i], 1000/nts[i],
                                              delta_r, lineshape)
    ca_T_err1[i] = ca_Res1[0][0][1] - ca_Res1[1][0][1]
    ca_T_err2[i] = ca_Res2[0][0][1] - ca_Res2[1][0][1]
    ca_T_err3[i] = ca_Res3[0][0][1] - ca_Res3[1][0][1]

    ca_plus_N_L = ca_plus_lib.N_L_from_pulse_energy(Es[i])
    ca_plus_Res1 = ca_plus_lib.get_wind_and_temp_errors(Temp_Ca, 
                                                       ca_plus_nu_Ls_errs,
                                                       ca_plus_Delta_nu_L,
                                                       ca_plus_N_L, z, T_atm,
                                                       alpha_L, alpha_T,
                                                       ca_plus_t_L, nts[i],
                                                       100/nts[i], delta_r,
                                                       lineshape)
    ca_plus_Res2 = ca_plus_lib.get_wind_and_temp_errors(Temp_Ca,
                                                        ca_plus_nu_Ls_errs,
                                                        ca_plus_Delta_nu_L/10,
                                                        ca_plus_N_L, z, T_atm,
                                                        alpha_L, alpha_T,
                                                        10*ca_plus_t_L, nts[i],
                                                        1000/nts[i], delta_r,
                                                        lineshape)
    ca_plus_Res3 = ca_plus_lib.get_wind_and_temp_errors(Temp_Ca,
                                                        ca_plus_nu_Ls_errs,
                                                        ca_plus_Delta_nu_L/10,
                                                        ca_plus_N_L, z, T_atm,
                                                        alpha_L2, alpha_T,
                                                        10*ca_plus_t_L, nts[i],
                                                        1000/nts[i], delta_r,
                                                        lineshape)
    ca_plus_T_err1[i] = ca_plus_Res1[0][0][1] - ca_plus_Res1[1][0][1]
    ca_plus_T_err2[i] = ca_plus_Res2[0][0][1] - ca_plus_Res2[1][0][1]
    ca_plus_T_err3[i] = ca_plus_Res3[0][0][1] - ca_plus_Res3[1][0][1]
    
Temp_bias_data = np.vstack((Es, ca_T_err1, ca_T_err2, ca_T_err3,
                            ca_plus_T_err1, ca_plus_T_err2, ca_plus_T_err3))
np.savetxt(os.path.join(outpath, 'Ca_and_ca_plus_temp_biases.txt'),
           Temp_bias_data.T, delimiter=',')

# Generate plot for Figure 2       
plt.figure(figsize=(12,8))

plt.plot(Es, ca_T_err1, label='Ca')
plt.plot(Es, ca_plus_T_err1, label='Ca+')
plt.legend()
plt.ylabel('Temperature bias (K)')
plt.xlabel('Pulse energy (mJ)')

plt.grid(True)

plt.savefig(os.path.join(outpath, 'Ca_and_ca_plus2.jpg'), dpi=300)

# Generate plot for Figure 4
plt.figure(figsize=(12,8))

plt.plot(Es, ca_T_err2, c='tab:blue', label='Ca')
plt.plot(Es, ca_T_err3, c='tab:blue', linestyle=':')
plt.plot(Es, ca_plus_T_err2, c='tab:orange', label='Ca+')
plt.plot(Es, ca_plus_T_err3, c='tab:orange', linestyle=':')
plt.legend()
plt.ylabel('Temperature bias (K)')
plt.xlabel('Pulse energy (mJ)')

plt.grid(True)

plt.savefig(os.path.join(outpath, 'Ca_and_ca_plus4.jpg'), dpi=300)
