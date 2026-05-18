# -*- coding: utf-8 -*-

import sys
sys.path.append("")
import iron_saturation_calculator_library as fe_lib
import os
import numpy as np
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

#Calculates the spectrum of the degree of saturation, following the Megie and
# VDG approaches.

nu_Ls = (np.arange(63) - 31)*1e8 #Hz
sats_vdG_gauss = np.zeros(len(nu_Ls))
sats_vdG_lorentz = np.zeros(len(nu_Ls))
sats_Megie = np.zeros(len(nu_Ls))

line = 2
E_pulse = 85 #mJ
t_L = 250 #ns
T_atm = 0.5
z = 90000 #km
alpha_L = 130e-6 #radians
alpha_T = 200e-6 #radians

nt = 250
delta_t = 5 #ns
delta_r = 50e-6 #radians

N_L = fe_lib.N_L_from_pulse_energy(E_pulse, line)
Delta_nu_L = 12e6 #Hz
Temp_Fe = 200 #K
Doppler_spectrum = fe_lib.get_total_scattering_cross_section_spectrum(Temp_Fe,
                                                                      line)

for i in range(len(nu_Ls)):
    print(i)
    nu_L = nu_Ls[i]    
    sats_vdG_gauss[i] = fe_lib.get_saturation_beam(line, nu_L, Delta_nu_L, N_L,
                        z, T_atm, alpha_L, alpha_T, t_L, nt, delta_t, delta_r,
                        Temp_Fe, 'gauss', ratio_beam=True)    
    sats_vdG_lorentz[i] = fe_lib.get_saturation_beam(line, nu_L, Delta_nu_L,
                        N_L, z, T_atm, alpha_L, alpha_T, t_L, nt, delta_t,
                        delta_r, Temp_Fe, 'lorentzian', ratio_beam=True)    
    g_L = fe_lib.get_laser_pulseshape(nu_L, Delta_nu_L, lineshape='gauss')
    sigma_eff = np.sum(g_L * Doppler_spectrum) / np.sum(g_L)
    sats_Megie[i] = fe_lib.get_saturation_megie(z, alpha_L, t_L, sigma_eff, N_L,
                                                T_atm, line)
    
#Calculates the saturation-induced density and temperature biases as a
# function of laser pulse energy

nu_Ls_errs = (np.arange(16) - 7.5) * 43e7 #Hz
line = 2 #Selects the 386 nm resonance line)

Es = 1.075 * 10**np.arange(0, 2.1, 0.2) #mJ (The factor 1.075 scales the 
                                        # maxiumum pulse energy such that the 
                                        # energy density is 1 mJ / m^2)
Es = np.hstack((1e-3, Es))

Omega = np.pi / 4 * np.sin(alpha_L)**2
u_0s = Es / z**2 / Omega

delta_ts = np.array([10, 10, 10, 10, 10, 10, 10, 10, 5, 5, 2, 2])
nts = 1250 / delta_ts
nts = nts.astype(np.int16)

dens_err_gauss = np.zeros(len(Es))
T_err_gauss = np.zeros(len(Es))
dens_err_lorentz = np.zeros(len(Es))
T_err_lorentz = np.zeros(len(Es))

for i in range(len(Es)):
    print(i)
    N_L = fe_lib.N_L_from_pulse_energy(Es[i], line)
    Res_gauss = fe_lib.get_wind_and_temp_errors(line, Temp_Fe, nu_Ls_errs,
                                Delta_nu_L, N_L, z, T_atm, alpha_L, alpha_T,
                                t_L, nts[i], delta_ts[i], delta_r, 'gauss')
    Res_lorentz = fe_lib.get_wind_and_temp_errors(line, Temp_Fe, nu_Ls_errs,
                                Delta_nu_L, N_L, z, T_atm, alpha_L, alpha_T,
                                t_L, nts[i], delta_ts[i], delta_r,
                                'lorentzian')
      
    dens_err_gauss[i] = ((Res_gauss[0][0][0] - Res_gauss[1][0][0]) 
                             / Res_gauss[1][0][0])
    T_err_gauss[i] = Res_gauss[0][0][1] - Res_gauss[1][0][1]
    
    dens_err_lorentz[i] = ((Res_lorentz[0][0][0] - Res_lorentz[1][0][0])
                                / Res_lorentz[1][0][0])
    T_err_lorentz[i] = Res_lorentz[0][0][1] - Res_lorentz[1][0][1]

fig,ax = plt.subplots(1,2, figsize=(16,8))

ax[0].plot(nu_Ls*1e-9, 100*sats_vdG_gauss,
           label='von der Gathen approach (Gauss profile)')
ax[0].plot(nu_Ls*1e-9, 100*sats_vdG_lorentz,
           label='von der Gathen approach (Lorentz profile)')
ax[0].plot(nu_Ls*1e-9, 100*sats_Megie,
           label=r'Megie approach ($\sigma_{\mathrm{t}}$)')

ax[0].text(.06,.92, '(a)', transform=plt.gcf().transFigure)
ax[0].set_ylabel('Saturation percent')
ax[0].set_xlabel('Laser frequency offset (GHz)')
ax[0].set_ylim(-1,38)
ax[0].legend(fontsize=16)
ax[0].grid(True)

ax[1].text(.51,.93, '(b)', transform=plt.gcf().transFigure)
a0, = ax[1].plot(u_0s, dens_err_gauss)
a1, = ax[1].plot(u_0s, dens_err_lorentz, c='tab:blue', linestyle='--')
a2, = ax[1].plot(u_0s, T_err_gauss)
a3, = ax[1].plot(u_0s, T_err_lorentz, c='tab:orange', linestyle='--')
ax[1].axvline(E_pulse / z**2 / Omega * T_atm, ls='--', c='k')
ax[1].set_ylim(-0.4,1.1)

ax[1].set_xlabel(r'Laser energy density (mJ/m$^2$)')
ax[1].set_ylabel('Fractional density bias \n Absolute temperature bias (K)')

ax[1].legend([(a0, a1), (a2,a3)],
             ['Density bias (Gauss/Lorentz profile)',
              'Temperature bias (Gauss/Lorentz profile)'],
           handler_map = {tuple : HandlerTupleVertical()}, fontsize=16)

ax[1].grid(True)
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
fig.tight_layout()

plt.savefig(os.path.join(outpath, 'Fe.pdf'), dpi=300)
