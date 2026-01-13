# -*- coding: utf-8 -*-

import sys
sys.path.append("")
import potassium_saturation_calculator_library as k_lib
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

font = {'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

c_light = 2.99792458 * 10**8 #m/s
lamb0 =  770.10835 * 10**-9 #m, in vacuum
nu0 = c_light / lamb0

#Designate an output location for figure data and plots
outpath_data = 'C:/Users/geac_ch/Documents/HELIX/Saturation paper/Fig_data/temp/'
outpath_figs = 'C:/Users/geac_ch/Documents/HELIX/Saturation paper/Figs/temp/'

#Calculates the intrinsic and effective spectra of the resonance lines. 
delta_nu = 1e6
nu_shifts = np.arange(-3*10**9, 3*10**9, delta_nu)
lambda_shifts = -nu_shifts/nu0 * lamb0

lines = []
for iso in range(2):
    for j in range(2):
        for k in range(2):
            lines.append(k_lib.get_natural_absorption_line(iso, j, k))

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

fig,ax = plt.subplots(1,2, figsize=(16,8))

ax[0].text(.06,.92, '(a)', transform=plt.gcf().transFigure)
ax[1].text(.51,.93, '(b)', transform=plt.gcf().transFigure)

ax[0].plot(lambda_shifts*1e12, lines[0], linestyle='-',
              label=r'$^{39}\alpha^1_1$', color=colors[0])
ax[0].plot(lambda_shifts*1e12, lines[1], linestyle='-',
              label=r'$^{39}\alpha^2_1$', color=colors[1])
ax[0].plot(lambda_shifts*1e12, lines[2], linestyle='-',
              label=r'$^{39}\alpha^1_2$', color=colors[2])
ax[0].plot(lambda_shifts*1e12, lines[3], linestyle='-',
              label=r'$^{39}\alpha^2_2$', color=colors[3])
ax[0].plot(lambda_shifts*1e12, lines[4], linestyle='--',
              label=r'$^{41}\alpha^1_1$', color=colors[0])
ax[0].plot(lambda_shifts*1e12, lines[5], linestyle='--',
              label=r'$^{41}\alpha^2_1$', color=colors[1])
ax[0].plot(lambda_shifts*1e12, lines[6], linestyle='--',
              label=r'$^{41}\alpha^1_2$', color=colors[2])
ax[0].plot(lambda_shifts*1e12, lines[7], linestyle='--',
              label=r'$^{41}\alpha^2_2$', color=colors[3])
ax[1].plot(lambda_shifts*1e12, k_lib.get_combined_absorption_line())
ax[0].legend(ncol=2)
for i in range(2):
    ax[i].set_xlabel('Wavelength (pm)')
    ax[i].set_xlim(-1.1,.6)
    ax[i].set_xticks([-1,-0.5,0,0.5])
ax[0].set_ylabel('Scattering cross-section (m$^2$)')
ax[1].set_ylabel('Effective scattering cross-section (m$^2$)')

fig.tight_layout()
plt.savefig(os.path.join(outpath_figs, 'K_spectrum.pdf'), dpi=300)
plt.show()

Data_K_spectrum_a = np.vstack((lambda_shifts*1e12, lines[0], lines[1], lines[2],
                            lines[3], lines[4], lines[5], lines[6], lines[7]))
np.savetxt(os.path.join(outpath_data, 'K_spectrum_a.txt'), Data_K_spectrum_a.T,
           delimiter=',')

Data_K_spectrum_b = np.vstack((lambda_shifts*1e12,
                               k_lib.get_combined_absorption_line()))
np.savetxt(os.path.join(outpath_data, 'K_spectrum_b.txt'), Data_K_spectrum_b.T,
           delimiter=',')

#Calculates the spectrum of the degree of saturation.   
nu_Ls = (np.arange(37) - 18)*1e8 #Hz
sats_lorentzian_nuL = np.zeros(len(nu_Ls))
sats_gauss_nuL = np.zeros(len(nu_Ls))

E_pulse = 100 #mJ
N_L = k_lib.N_L_from_pulse_energy(E_pulse)
t_L = 275 #ns
T_atm = 0.7
Temp_K = 200 #K
z = 90000 #km
alpha_L = 133e-6 #radians
alpha_T = 186e-6 #radians
delta_r = 50e-6 #radians

nt = 1000
delta_t = 1.25 #ns

Delta_nu_L = 20e6 #Hz

for i in range(len(nu_Ls)):
    print(i)
    nu_L = nu_Ls[i]    
    sats_lorentzian_nuL[i] = k_lib.get_saturation_beam(nu_L, Delta_nu_L, N_L, z,
                       T_atm, alpha_L, alpha_T, t_L, nt, delta_t, delta_r,
                       Temp_K, 'lorentzian', ratio_beam=True)
    sats_gauss_nuL[i] = k_lib.get_saturation_beam(nu_L, Delta_nu_L, N_L, z,
                       T_atm, alpha_L, alpha_T, t_L, nt, delta_t, delta_r,
                       Temp_K, 'gauss', ratio_beam=True)

#Calculates the degree of saturation as a function of laser pulse energy.      
Es = 10**np.arange(0, 2.1, 0.2) #mJ
Es = np.hstack((1e-3, Es))

sats_lorentzian_E = np.zeros(len(Es)) 
sats_gauss_E = np.zeros(len(Es))

for i in range(len(Es)):
    print(i)
    N_L = k_lib.N_L_from_pulse_energy(Es[i])
    sats_lorentzian_E[i] = k_lib.get_saturation_beam(0, Delta_nu_L, N_L, z,
                          T_atm, alpha_L, alpha_T, t_L, nt, delta_t, delta_r,
                          Temp_K, 'lorentzian', ratio_beam=True)
    sats_gauss_E[i] = k_lib.get_saturation_beam(0, Delta_nu_L, N_L, z, T_atm,
                          alpha_L, alpha_T, t_L, nt, delta_t, delta_r, Temp_K,
                          'gauss', ratio_beam=True)
      
lambda_Ls = -nu_Ls/nu0 * lamb0
    
fig,ax = plt.subplots(1,2, figsize=(16,8))

ax[0].plot(lambda_Ls*1e12, 100*sats_lorentzian_nuL,
           label='Lorentzian profile')
ax[0].plot(lambda_Ls*1e12, 100*sats_gauss_nuL, label='Gaussian profile')

ax[0].text(.06,.92, '(a)', transform=plt.gcf().transFigure)
ax[0].set_ylabel('Saturation percent')
ax[0].set_xlabel('Central laser wavelength (pm)')
ax[0].set_ylim(-4,87)
ax[0].legend()
ax[0].grid(True)

ax[1].text(.51,.93, '(b)', transform=plt.gcf().transFigure)
ax[1].plot(Es, 100*sats_lorentzian_E, label = 'Lorentzian profile')
ax[1].plot(Es, 100*sats_gauss_E, label = 'Gaussian profile')

ax[1].set_xlabel('Laser pulse energy (mJ)')
ax[1].set_ylabel('Saturation percent')
ax[1].legend()
ax[1].grid(True)
fig.tight_layout()

plt.savefig(os.path.join(outpath_figs, 'K_saturation.pdf'), dpi=300)
plt.show()

Data_K_saturation_a = np.vstack((lambda_Ls*1e12, 100*sats_lorentzian_nuL,
                                 100*sats_gauss_nuL))
np.savetxt(os.path.join(outpath_data, 'K_saturation_a.txt'),
           Data_K_saturation_a.T, delimiter=',')

Data_K_saturation_b = np.vstack((Es, 100*sats_lorentzian_E, 100*sats_gauss_E))
np.savetxt(os.path.join(outpath_data, 'K_saturation_b.txt'),
           Data_K_saturation_b.T, delimiter=',')

#Calculates the saturation-induced temperature and wind errors as a function of
#  laser pulse energy.      
Es = 10**np.arange(0, 2.1, 0.2) #mJ
Es = np.hstack((1e-3, Es))
T_err_200_lorentzian = np.zeros(len(Es)) #K
w_err_200_lorentzian = np.zeros(len(Es)) #m/s
T_err_200_gauss = np.zeros(len(Es)) #K
w_err_200_gauss = np.zeros(len(Es)) #m/s
T_err_150_lorentzian = np.zeros(len(Es)) #K
w_err_150_lorentzian = np.zeros(len(Es)) #m/s
T_err_150_gauss = np.zeros(len(Es)) #K
w_err_150_gauss = np.zeros(len(Es)) #m/s

alpha_L = 270e-6 #radians

nt = 3000
delta_t = 50 #ns

lambda_Ls = np.arange(1.55, -1.52, -0.18)*1e-12
nu_Ls = -c_light / lamb0**2 * lambda_Ls  

for i in range(len(Es)):
    print(i)
    delta_t = min(1.5 * 100 / Es[i], 10)
    N_L = k_lib.N_L_from_pulse_energy(Es[i])
    Res_lorentzian_200 = k_lib.get_wind_and_temp_errors(200, nu_Ls, Delta_nu_L,
                                      N_L, z, T_atm, alpha_L, alpha_T, t_L, nt,
                                      delta_t, delta_r, 'lorentzian')
     
    Res_gauss_200 = k_lib.get_wind_and_temp_errors(200, nu_Ls, Delta_nu_L, N_L,
                                      z, T_atm, alpha_L, alpha_T, t_L, nt,
                                      delta_t, delta_r, 'gauss')
    
    Res_lorentzian_150 = k_lib.get_wind_and_temp_errors(150, nu_Ls, Delta_nu_L,
                                      N_L, z, T_atm, alpha_L, alpha_T, t_L, nt,
                                      delta_t, delta_r, 'lorentzian')
     
    Res_gauss_150 = k_lib.get_wind_and_temp_errors(150, nu_Ls, Delta_nu_L, N_L,
                                      z, T_atm, alpha_L, alpha_T, t_L, nt,
                                      delta_t, delta_r, 'gauss')
      
    
    T_err_200_lorentzian[i] = (Res_lorentzian_200[0][0][1] - 
                                              Res_lorentzian_200[1][0][1])
    w_err_200_lorentzian[i] = (Res_lorentzian_200[0][0][2] -
                                              Res_lorentzian_200[1][0][2])
    
    T_err_200_gauss[i] = Res_gauss_200[0][0][1] - Res_gauss_200[1][0][1]
    w_err_200_gauss[i] = Res_gauss_200[0][0][2] - Res_gauss_200[1][0][2]
    
    
    T_err_150_lorentzian[i] = (Res_lorentzian_150[0][0][1] -
                                          Res_lorentzian_150[1][0][1])
    w_err_150_lorentzian[i] = (Res_lorentzian_150[0][0][2] -
                                         Res_lorentzian_150[1][0][2])
    
    T_err_150_gauss[i] = Res_gauss_150[0][0][1] - Res_gauss_150[1][0][1]
    w_err_150_gauss[i] = Res_gauss_150[0][0][2] - Res_gauss_150[1][0][2]  

fig,ax = plt.subplots(2,2, figsize=(16,16))

ax[0,0].text(.06,.9, '(a)', transform=plt.gcf().transFigure)
ax[0,0].text(.53,.9, '(b)', transform=plt.gcf().transFigure)
ax[0,0].text(.06,.49, '(c)', transform=plt.gcf().transFigure)
ax[0,0].text(.53,.49, '(d)', transform=plt.gcf().transFigure)
ax[0,0].plot(Es, T_err_150_lorentzian, label = 'Lorentzian profile')
ax[0,0].plot(Es, T_err_150_gauss, label = 'Gaussian profile')
ax[0,0].set_ylim(-21,1)
ax[0,0].set_yticks([-20,-15,-10,-5,0])
ax[0,0].grid(True)
ax[0,0].legend()
ax[0,0].set_ylabel('Temperature bias (K)')

ax[1,0].plot(Es, w_err_150_lorentzian, label = 'Lorentzian profile')
ax[1,0].plot(Es, w_err_150_gauss, label = 'Gaussian profile')
ax[1,0].set_ylim(-32,1.5)
ax[1,0].set_xlabel('Laser pulse energy (mJ)')
ax[1,0].set_ylabel('Line-of-sight wind bias (m/s)')
ax[1,0].grid(True)

ax[0,1].plot(Es, T_err_200_lorentzian, label = 'Lorentzian profile')
ax[0,1].plot(Es, T_err_200_gauss, label = 'Gaussian profile')
ax[0,1].set_ylim(-21,1)
ax[0,1].set_yticks([-20,-15,-10,-5,0])
ax[0,1].grid(True)

ax[1,1].plot(Es, w_err_200_lorentzian, label = 'Lorentzian profile')
ax[1,1].plot(Es, w_err_200_gauss, label = 'Gaussian profile')
ax[1,1].set_ylim(-32,1.5)
ax[1,1].set_xlabel('Laser pulse energy (mJ)')
ax[1,1].grid(True)

plt.savefig(os.path.join(outpath_figs, 'K_temp_and_wind_biases.pdf'), dpi=300)
plt.show()

Data_K_measurements_a = np.vstack((Es, T_err_150_lorentzian, T_err_150_gauss))
np.savetxt(os.path.join(outpath_data, 'K_temp_and_wind_biases_a.txt'),
           Data_K_measurements_a.T, delimiter=',')

Data_K_measurements_b = np.vstack((Es, T_err_200_lorentzian, T_err_200_gauss))
np.savetxt(os.path.join(outpath_data, 'K_temp_and_wind_biases_b.txt'),
           Data_K_measurements_b.T, delimiter=',')

Data_K_measurements_c = np.vstack((Es, w_err_150_lorentzian, w_err_150_gauss))
np.savetxt(os.path.join(outpath_data, 'K_temp_and_wind_biases_c.txt'),
           Data_K_measurements_c.T, delimiter=',')

Data_K_measurements_d = np.vstack((Es, w_err_200_lorentzian, w_err_200_gauss))
np.savetxt(os.path.join(outpath_data, 'K_temp_and_wind_biases_d.txt'),
           Data_K_measurements_d.T, delimiter=',')