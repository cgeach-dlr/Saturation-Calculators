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
      
outpath = os.path.join(os.path.dirname(os.getcwd()), 'Output')

Data_Fe_saturation_fname = os.path.join(outpath, 'Figure Data',
                                     'Fe_saturation.txt')
Data_Fe_saturation = np.loadtxt(Data_Fe_saturation_fname,
                        delimiter=',').T
Data_Fe_biases_fname = os.path.join(outpath, 'Figure Data',
                                     'Fe_temp_and_wind_biases.txt')
Data_Fe_biases = np.loadtxt(Data_Fe_biases_fname,
                        delimiter=',').T

E_pulse = 85 #mJ
z = 90000 #m
alpha_L = 130e-6 #rad
Omega = np.pi / 4 * np.sin(alpha_L)**2
T_atm = 0.5
                        
nu_Ls = Data_Fe_saturation[0]
sats_vdG_gauss = Data_Fe_saturation[1]
sats_vdG_lorentz = Data_Fe_saturation[2]
sats_Megie = Data_Fe_saturation[3]

u_0s = Data_Fe_biases[0]
dens_err_200_gauss = Data_Fe_biases[1]
dens_err_200_lorentz = Data_Fe_biases[2]
T_err_200_gauss = Data_Fe_biases[3]
T_err_200_lorentz = Data_Fe_biases[4]

fig,ax = plt.subplots(1,2, figsize=(16,8))

ax[0].plot(nu_Ls, sats_vdG_gauss, label='von der Gathen ' + 
           'approach (Gauss profile)')
ax[0].plot(nu_Ls, sats_vdG_lorentz, label='von der Gathen ' +
           'approach (Lorentz profile)')
ax[0].plot(nu_Ls, sats_Megie, label='Megie approach ' + 
           r'($\sigma_{\mathrm{t}}$)')

ax[0].text(.06,.92, '(a)', transform=plt.gcf().transFigure)
ax[0].set_ylabel('Saturation percent')
ax[0].set_xlabel('Laser frequency offset (GHz)')
ax[0].set_ylim(-1,38)
ax[0].legend(fontsize=16)
ax[0].grid(True)

ax[1].text(.51,.93, '(b)', transform=plt.gcf().transFigure)
a0, = ax[1].plot(u_0s, dens_err_200_gauss)
a1, = ax[1].plot(u_0s, dens_err_200_lorentz, c='tab:blue', linestyle='--')
a2, = ax[1].plot(u_0s, T_err_200_gauss)
a3, = ax[1].plot(u_0s, T_err_200_lorentz, c='tab:orange', linestyle='--')
ax[1].axvline(E_pulse / z**2 / Omega * T_atm, ls='--', c='k')
ax[1].set_ylim(-0.4,1.1)

ax[1].set_xlabel(r'Laser energy density (mJ/m$^2$)')
ax[1].set_ylabel('Fractional density bias \n Absolute temperature bias (K)')

ax[1].legend([(a0, a1), (a2,a3)],
             ['Density bias (Gauss/Lorentz profile)',
              'Temperature bias (Gauss/Lorentz profile)'],
           handler_map = {tuple : HandlerTupleVertical()},
           fontsize=16, framealpha=1)

ax[1].grid(True)
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
fig.tight_layout()

plt.savefig(os.path.join(outpath, 'Fe.pdf'), dpi=300)
