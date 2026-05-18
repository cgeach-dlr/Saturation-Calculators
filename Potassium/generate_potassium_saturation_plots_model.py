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

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

from matplotlib.legend_handler import HandlerTuple

class HandlerTupleVertical(HandlerTuple):
    """Plots all the given Lines vertical stacked."""

    def __init__(self, **kwargs):
        """Run Base Handler."""
        HandlerTuple.__init__(self, **kwargs)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        """Create artists (the symbol) for legend entry."""
        # How many lines are there.
        numlines = len(orig_handle)
        handler_map = legend.get_legend_handler_map()

        # divide the vertical space where the lines will go
        # into equal parts based on the number of lines
        height_y = (height / numlines)

        leglines = []
        for i, handle in enumerate(orig_handle):
            handler = legend.get_legend_handler(handler_map, handle)

            legline = handler.create_artists(legend, handle,
                                             xdescent,
                                             (2*i + 1)*height_y,
                                             width,
                                             2*height,
                                             fontsize, trans)
            leglines.extend(legline)

        return leglines


#Designate an output location for figure data and plots
outpath = os.path.join(os.path.dirname(os.getcwd()), 'Output')

#Calculates the intrinsic and effective spectra of the resonance lines. 
lambda_shifts = -k_lib.nu_shifts / k_lib.nu0 * k_lib.lamb0

lines = []
for iso in range(2):
    for j in range(2):
        for k in range(2):
            lines.append(k_lib.get_natural_absorption_line(iso, j, k))

combined_spectrum = k_lib.get_combined_absorption_line()
doppler_spectrum = k_lib.get_doppler_broadened_spectrum_complete(200)

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
ax[1].plot(lambda_shifts*1e12, combined_spectrum,
           label='Homogeneous broadening')
ax[1].plot(lambda_shifts*1e12, 100*doppler_spectrum, 'k--',
           label='Homogeneous + inhomogeneous \n broadening at 200 K, x100')
ax[0].legend(ncol=2)
ax[1].legend()
for i in range(2):
  ax[i].set_xlabel('Wavelength offset (pm)')
ax[0].set_xlim(-1.1,1.1)
ax[1].set_xlim(-2.1,2.1)
ax[1].set_ylim(-5e-15, 13e-14)
ax[0].set_ylabel('Instrinsic scattering cross-section (m$^2$)')
ax[1].set_ylabel('Effective scattering cross-section (m$^2$)')

fig.tight_layout()
Fig_K_spectrum_fname = os.path.join(outpath, 'K_spectrum.pdf')
plt.savefig(Fig_K_spectrum_fname, dpi=300)
plt.show()

#Calculates the spectrum of the degree of saturation.   
nu_Ls = (np.arange(37) - 18)*1e8 #Hz
sats_gauss_nuL = np.zeros(len(nu_Ls))
sats_lorentz_nuL = np.zeros(len(nu_Ls))
sats_Megie_nuL = np.zeros(len(nu_Ls))

E_pulse = 100 #mJ
N_L = k_lib.N_L_from_pulse_energy(E_pulse)
t_L = 275 #ns
T_atm = 0.7
Temp_K = 200 #K
z = 92500 #km
alpha_L = 133e-6 #radians
alpha_T = 186e-6 #radians
delta_r = 50e-6 #radians
Delta_nu_L = 20e6 #Hz
Doppler_spectrum = k_lib.get_total_scattering_cross_section_spectrum(Temp_K)

nt = 1000
delta_t = 1.25 #ns

for i in range(len(nu_Ls)):
    nu_L = nu_Ls[i]    
    sats_gauss_nuL[i] = k_lib.get_saturation_beam(nu_L, Delta_nu_L, N_L, z, 
                                                  T_atm, alpha_L, alpha_T, t_L,
                                                  nt, delta_t, delta_r, Temp_K,
                                                  'gauss', ratio_beam=True)
    sats_lorentz_nuL[i] = k_lib.get_saturation_beam(nu_L, Delta_nu_L, N_L, z,
                                                T_atm, alpha_L, alpha_T, t_L,
                                                nt, delta_t, delta_r, Temp_K,
                                                'lorentzian', ratio_beam=True)
    g_L = k_lib.get_laser_pulseshape(nu_L, Delta_nu_L, 'gauss')
    sigma_eff = np.sum(g_L * Doppler_spectrum) / np.sum(g_L)
    sats_Megie_nuL[i] = k_lib.get_saturation_megie(z, alpha_L, t_L, sigma_eff,
                                                 N_L, T_atm)

#Calculates the degree of saturation as a function of laser pulse energy.      
Es = 10**np.arange(0, 2.1, 0.2) #mJ
Es = np.hstack((1e-3, Es))

sats_gauss_E = np.zeros(len(Es))
sats_lorentz_E = np.zeros(len(Es)) 
sats_Megie_E = np.zeros(len(Es)) 

for i in range(len(Es)):
    print(i)
    N_L = k_lib.N_L_from_pulse_energy(Es[i])
    sats_gauss_E[i] = k_lib.get_saturation_beam(0, Delta_nu_L, N_L, z, T_atm,
                                                alpha_L, alpha_T, t_L, nt,
                                                delta_t, delta_r, Temp_K,
                                                'gauss', ratio_beam=True)
    sats_lorentz_E[i] = k_lib.get_saturation_beam(0, Delta_nu_L, N_L, z,
                                                T_atm, alpha_L, alpha_T, t_L,
                                                nt, delta_t, delta_r, Temp_K,
                                                'lorentzian', ratio_beam=True)
    g_L = k_lib.get_laser_pulseshape(0, Delta_nu_L, 'gauss')
    sigma_eff = np.sum(g_L * Doppler_spectrum) / np.sum(g_L)
    sats_Megie_E[i] = k_lib.get_saturation_megie(z, alpha_L, t_L, sigma_eff,
                                                 N_L, T_atm)

lambda_Ls = -nu_Ls / k_lib.nu0 * k_lib.lamb0
    
fig,ax = plt.subplots(1,2, figsize=(16,8))

ax[0].plot(lambda_Ls*1e12, 100*sats_gauss_nuL, label='Gaussian profile')
ax[0].plot(lambda_Ls*1e12, 100*sats_lorentz_nuL,
           label='Lorentzian profile')
ax[0].plot(lambda_Ls*1e12, 100*sats_Megie_nuL, label='Megie approach')

ax[0].text(.06,.92, '(a)', transform=plt.gcf().transFigure)
ax[0].set_ylabel('Saturation percent')
ax[0].set_xlabel('Wavelength offset (pm)')
ax[0].set_ylim(-4,87)
ax[0].legend()
ax[0].grid(True)

ax[1].text(.51,.93, '(b)', transform=plt.gcf().transFigure)
ax[1].plot(Es, 100*sats_gauss_E, label = 'Gaussian profile')
ax[1].plot(Es, 100*sats_lorentz_E, label = 'Lorentzian profile')
ax[1].plot(Es, 100*sats_Megie_E, label = 'Megie approach')

ax[1].set_xlabel('Laser pulse energy (mJ)')
ax[1].set_ylabel('Saturation percent')
ax[1].legend()
ax[1].grid(True)
fig.tight_layout()

Fig_K_saturation_fname = os.path.join(outpath, 'K_saturation.pdf')
plt.savefig(Fig_K_saturation_fname, dpi=300)
plt.show()

t_L = 275 #ns
T_atm = 0.7
z = 92500 #km

Delta_nu_L = 20e6 #Hz

#Calculates the saturation-induced temperature and wind errors as a function of
# laser pulse energy.      
Es = 10**np.arange(0, 2.1, 0.2) #mJ
Es = np.hstack((1e-3, Es))

T_err_200_lorentz = np.zeros(len(Es)) #K
w_err_200_lorentz = np.zeros(len(Es)) #m/s
T_err_200_gauss = np.zeros(len(Es)) #K
w_err_200_gauss = np.zeros(len(Es)) #m/s
T_err_150_lorentz = np.zeros(len(Es)) #K
w_err_150_lorentz = np.zeros(len(Es)) #m/s
T_err_150_gauss = np.zeros(len(Es)) #K
w_err_150_gauss = np.zeros(len(Es)) #m/s

T_err_200_lorentz_noise = np.zeros(len(Es)) #K
w_err_200_lorentz_noise = np.zeros(len(Es)) #m/s
T_err_200_gauss_noise = np.zeros(len(Es)) #K
w_err_200_gauss_noise = np.zeros(len(Es)) #m/s
T_err_150_lorentz_noise = np.zeros(len(Es)) #K
w_err_150_lorentz_noise = np.zeros(len(Es)) #m/s
T_err_150_gauss_noise = np.zeros(len(Es)) #K
w_err_150_gauss_noise = np.zeros(len(Es)) #m/s

alpha_L = 270e-6 #radians
alpha_T = 186e-6 #radians
delta_r = 50e-6 #radians

nt = 3000
delta_t = 50 #ns

lambda_Ls_errs = np.arange(1.55, -1.52, -0.18)*1e-12
nu_Ls_errs = -k_lib.c_light / k_lib.lamb0**2 * lambda_Ls_errs  

for i in range(len(Es)):
    delta_t = min(1.5 * 100 / Es[i], 10)
    N_L = k_lib.N_L_from_pulse_energy(Es[i])
    Res_gauss_200 = k_lib.get_wind_and_temp_errors(200, nu_Ls_errs, Delta_nu_L,
                                                   N_L, z, T_atm, alpha_L,
                                                   alpha_T, t_L, nt, delta_t,
                                                   delta_r, 'gauss',
                                                   False)
    Res_lorentz_200 = k_lib.get_wind_and_temp_errors(200, nu_Ls_errs, 
                                                     Delta_nu_L, N_L, z, T_atm,
                                                     alpha_L, alpha_T, t_L, nt,
                                                     delta_t, delta_r,
                                                     'lorentzian', False)
    Res_gauss_150 = k_lib.get_wind_and_temp_errors(150, nu_Ls_errs, Delta_nu_L,
                                                   N_L, z, T_atm, alpha_L, 
                                                   alpha_T, t_L, nt, delta_t,
                                                   delta_r, 'gauss', False)
    Res_lorentz_150 = k_lib.get_wind_and_temp_errors(150, nu_Ls_errs,
                                                     Delta_nu_L, N_L, z, T_atm,
                                                     alpha_L, alpha_T, t_L, nt,
                                                     delta_t, delta_r,
                                                     'lorentzian', False)
    
    Res_gauss_200_noise = k_lib.get_wind_and_temp_errors(200, nu_Ls_errs,
                                                         Delta_nu_L, N_L, z,
                                                         T_atm, alpha_L, 
                                                         alpha_T, t_L, nt,
                                                         delta_t, delta_r,
                                                         'gauss', True)
    Res_lorentz_200_noise = k_lib.get_wind_and_temp_errors(200, nu_Ls_errs,
                                                           Delta_nu_L, N_L, z,
                                                           T_atm, alpha_L,
                                                           alpha_T, t_L, nt,
                                                           delta_t, delta_r,
                                                           'lorentzian', True)
    Res_gauss_150_noise = k_lib.get_wind_and_temp_errors(150, nu_Ls_errs,
                                                         Delta_nu_L, N_L, z,
                                                         T_atm, alpha_L, 
                                                         alpha_T, t_L, nt,
                                                         delta_t, delta_r,
                                                         'gauss', True)
    Res_lorentz_150_noise = k_lib.get_wind_and_temp_errors(150, nu_Ls_errs,
                                                           Delta_nu_L, N_L, z,
                                                           T_atm, alpha_L,
                                                           alpha_T, t_L, nt,
                                                           delta_t, delta_r,
                                                           'lorentzian', True)
        
    T_err_200_lorentz[i] = Res_lorentz_200[0][0][1] - Res_lorentz_200[1][0][1]
    w_err_200_lorentz[i] = Res_lorentz_200[0][0][2] - Res_lorentz_200[1][0][2]
    
    T_err_200_gauss[i] = Res_gauss_200[0][0][1] - Res_gauss_200[1][0][1]
    w_err_200_gauss[i] = Res_gauss_200[0][0][2] - Res_gauss_200[1][0][2]
    
    
    T_err_150_lorentz[i] = Res_lorentz_150[0][0][1] - Res_lorentz_150[1][0][1]
    w_err_150_lorentz[i] = Res_lorentz_150[0][0][2] - Res_lorentz_150[1][0][2]
    
    T_err_150_gauss[i] = Res_gauss_150[0][0][1] - Res_gauss_150[1][0][1]
    w_err_150_gauss[i] = Res_gauss_150[0][0][2] - Res_gauss_150[1][0][2]  
    
    T_err_200_lorentz_noise[i] = (Res_lorentz_200_noise[0][0][1]
                                  - Res_lorentz_200_noise[1][0][1])
    w_err_200_lorentz_noise[i] = (Res_lorentz_200_noise[0][0][2]
                                  - Res_lorentz_200_noise[1][0][2])
    
    T_err_200_gauss_noise[i] = (Res_gauss_200_noise[0][0][1]
                                - Res_gauss_200_noise[1][0][1])
    w_err_200_gauss_noise[i] = (Res_gauss_200_noise[0][0][2]
                                - Res_gauss_200_noise[1][0][2])
    
    
    T_err_150_lorentz_noise[i] = (Res_lorentz_150_noise[0][0][1]
                                  - Res_lorentz_150_noise[1][0][1])
    w_err_150_lorentz_noise[i] = (Res_lorentz_150_noise[0][0][2]
                                  - Res_lorentz_150_noise[1][0][2])
    
    T_err_150_gauss_noise[i] = (Res_gauss_150_noise[0][0][1]
                                - Res_gauss_150_noise[1][0][1])
    w_err_150_gauss_noise[i] = (Res_gauss_150_noise[0][0][2]
                                - Res_gauss_150_noise[1][0][2])

fig,ax = plt.subplots(2,2, figsize=(16,16))

ax[0,0].text(.06,.9, '(a)', transform=plt.gcf().transFigure)
ax[0,0].text(.53,.9, '(b)', transform=plt.gcf().transFigure)
ax[0,0].text(.06,.49, '(c)', transform=plt.gcf().transFigure)
ax[0,0].text(.53,.49, '(d)', transform=plt.gcf().transFigure)

a0, = ax[0,0].plot(Es, T_err_150_gauss)
a1, = ax[0,0].plot(Es, T_err_150_gauss_noise, c='tab:blue', linestyle='--')
a2, = ax[0,0].plot(Es, T_err_150_lorentz)
a3, = ax[0,0].plot(Es, T_err_150_lorentz_noise, c='tab:orange', linestyle='--')
ax[0,0].set_ylim(-21,11)
ax[0,0].set_yticks([-20,-15,-10,-5,0,5,10])
ax[0,0].grid(True)
ax[0,0].set_ylabel('Temperature bias (K)')
ax[0,0].legend([(a0, a1), (a2,a3)],
             ['Gaussian profile',
              'Lorentzian profile'],
           handler_map = {tuple : HandlerTupleVertical()}, fontsize=16)

ax[1,0].plot(Es, w_err_150_gauss)
ax[1,0].plot(Es, w_err_150_gauss_noise, c='tab:blue', linestyle='--')
ax[1,0].plot(Es, w_err_150_lorentz)
ax[1,0].plot(Es, w_err_150_lorentz_noise, c='tab:orange', linestyle='--')
ax[1,0].set_ylim(-32,1.5)
ax[1,0].set_xlabel('Laser pulse energy (mJ)')
ax[1,0].set_ylabel('Line-of-sight wind bias (m/s)')
ax[1,0].grid(True)

ax[0,1].plot(Es, T_err_200_gauss, label = 'Gaussian profile')
ax[0,1].plot(Es, T_err_200_gauss_noise, c='tab:blue', linestyle='--',
             label = 'Gaussian profile')
ax[0,1].plot(Es, T_err_200_lorentz, label = 'Lorentzian profile')
ax[0,1].plot(Es, T_err_200_lorentz_noise, c='tab:orange', linestyle='--',
             label = 'Lorentzian profile')

ax[0,1].set_ylim(-21,11)
ax[0,1].set_yticks([-20,-15,-10,-5,0,5,10])
ax[0,1].grid(True)

ax[1,1].plot(Es, w_err_200_gauss)
ax[1,1].plot(Es, w_err_200_gauss_noise, c='tab:blue', linestyle='--')
ax[1,1].plot(Es, w_err_200_lorentz)
ax[1,1].plot(Es, w_err_200_lorentz_noise, c='tab:orange', linestyle='--')
ax[1,1].set_ylim(-32,1.5)
ax[1,1].set_xlabel('Laser pulse energy (mJ)')
ax[1,1].grid(True)

Fig_K_biases_fname = os.path.join(outpath, 'K_biases.pdf')
plt.savefig(Fig_K_biases_fname, dpi=300)
plt.show()
