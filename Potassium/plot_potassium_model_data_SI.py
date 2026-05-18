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

Data_K_sats_nuL_fname = os.path.join(outpath, 'Figure Data', 
                                     'K_saturation_nuL_SI.txt')
Data_K_sats_nuL = np.loadtxt(Data_K_sats_nuL_fname, delimiter=',').T
Data_K_sats_E_fname = os.path.join(outpath, 'Figure Data', 
                                   'K_saturation_E_SI.txt')
Data_K_sats_E = np.loadtxt(Data_K_sats_E_fname, delimiter=',').T

lambda_Ls = Data_K_sats_nuL[0]
sats_gauss_nuL = Data_K_sats_nuL[1] 
sats_lorentz_nuL = Data_K_sats_nuL[2]
sats_Megie_nuL = Data_K_sats_nuL[3]

Es = Data_K_sats_E[0] 
sats_gauss_E = Data_K_sats_E[1] 
sats_lorentz_E = Data_K_sats_E[2] 
sats_Megie_E = Data_K_sats_E[3] 

fig,ax = plt.subplots(1,2, figsize=(16,8))

ax[0].plot(lambda_Ls, sats_gauss_nuL, label='Gaussian profile')
ax[0].plot(lambda_Ls, sats_lorentz_nuL,
           label='Lorentzian profile')
ax[0].plot(lambda_Ls, sats_Megie_nuL, label='Megie approach')

ax[0].text(.06,.92, '(a)', transform=plt.gcf().transFigure)
ax[0].set_ylabel('Saturation percent')
ax[0].set_xlabel('Wavelength offset (pm)')
ax[0].set_ylim(-4,68)
#ax[0].legend()
ax[0].grid(True)

ax[1].text(.51,.93, '(b)', transform=plt.gcf().transFigure)
ax[1].plot(Es, sats_gauss_E, label = 'Gaussian profile')
ax[1].plot(Es, sats_lorentz_E, label = 'Lorentzian profile')
ax[1].plot(Es, sats_Megie_E, label = 'Megie approach')

ax[1].set_xlabel('Laser pulse energy (mJ)')
ax[1].set_ylabel('Saturation percent')
ax[1].legend()
ax[1].grid(True)
fig.tight_layout()

Fig_K_saturation_fname = os.path.join(outpath, 'K_saturation_SI.pdf')
plt.savefig(Fig_K_saturation_fname, dpi=300)
plt.show()

Data_K_T_biases_150_fname = os.path.join(outpath, 'Figure Data', 
                                         'K_T_biases_150_SI.txt')
Data_K_w_biases_150_fname = os.path.join(outpath, 'Figure Data', 
                                         'K_w_biases_150_SI.txt')
Data_K_T_biases_200_fname = os.path.join(outpath, 'Figure Data', 
                                         'K_T_biases_200_SI.txt')
Data_K_w_biases_200_fname = os.path.join(outpath, 'Figure Data', 
                                         'K_w_biases_200_SI.txt')

Data_K_T_biases_150 = np.loadtxt(Data_K_T_biases_150_fname,
                                 delimiter=',').T
Data_K_w_biases_150 = np.loadtxt(Data_K_w_biases_150_fname,
                                 delimiter=',').T
Data_K_T_biases_200 = np.loadtxt(Data_K_T_biases_200_fname,
                                 delimiter=',').T
Data_K_w_biases_200 = np.loadtxt(Data_K_w_biases_200_fname,
                                 delimiter=',').T

Es = Data_K_T_biases_150[0]

T_err_150_gauss = Data_K_T_biases_150[1]
T_err_150_gauss_noise = Data_K_T_biases_150[2]
T_err_150_lorentz = Data_K_T_biases_150[3]
T_err_150_lorentz_noise = Data_K_T_biases_150[4]

w_err_150_gauss = Data_K_w_biases_150[1]
w_err_150_gauss_noise = Data_K_w_biases_150[2]
w_err_150_lorentz = Data_K_w_biases_150[3]
w_err_150_lorentz_noise = Data_K_w_biases_150[4]

T_err_200_gauss = Data_K_T_biases_200[1]
T_err_200_gauss_noise = Data_K_T_biases_200[2]
T_err_200_lorentz = Data_K_T_biases_200[3]
T_err_200_lorentz_noise = Data_K_T_biases_200[4]

w_err_200_gauss = Data_K_w_biases_200[1]
w_err_200_gauss_noise = Data_K_w_biases_200[2]
w_err_200_lorentz = Data_K_w_biases_200[3]
w_err_200_lorentz_noise = Data_K_w_biases_200[4]

fig,ax = plt.subplots(2,2, figsize=(16,16))

ax[0,0].text(.06,.9, '(a)', transform=plt.gcf().transFigure)
ax[0,0].text(.53,.9, '(b)', transform=plt.gcf().transFigure)
ax[0,0].text(.06,.49, '(c)', transform=plt.gcf().transFigure)
ax[0,0].text(.53,.49, '(d)', transform=plt.gcf().transFigure)

a0, = ax[0,0].plot(Es, T_err_150_gauss)
a1, = ax[0,0].plot(Es, T_err_150_gauss_noise, c='tab:blue', linestyle='--')
a2, = ax[0,0].plot(Es, T_err_150_lorentz)
a3, = ax[0,0].plot(Es, T_err_150_lorentz_noise, c='tab:orange', linestyle='--')
ax[0,0].set_ylim(-1,9)
ax[0,0].set_yticks([0,2,4,6,8])
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
ax[1,0].set_ylim(-3,.5)
ax[1,0].set_xlabel('Laser pulse energy (mJ)')
ax[1,0].set_ylabel('Line-of-sight wind bias (m/s)')
ax[1,0].grid(True)

ax[0,1].plot(Es, T_err_200_gauss, label = 'Gaussian profile')
ax[0,1].plot(Es, T_err_200_gauss_noise, c='tab:blue', linestyle='--',
             label = 'Gaussian profile')
ax[0,1].plot(Es, T_err_200_lorentz, label = 'Lorentzian profile')
ax[0,1].plot(Es, T_err_200_lorentz_noise, c='tab:orange', linestyle='--',
             label = 'Lorentzian profile')

ax[0,1].set_ylim(-1,9)
ax[0,1].set_yticks([0,2,4,6,8])
ax[0,1].grid(True)

ax[1,1].plot(Es, w_err_200_gauss)
ax[1,1].plot(Es, w_err_200_gauss_noise, c='tab:blue', linestyle='--')
ax[1,1].plot(Es, w_err_200_lorentz)
ax[1,1].plot(Es, w_err_200_lorentz_noise, c='tab:orange', linestyle='--')
ax[1,1].set_ylim(-3,.5)
ax[1,1].set_xlabel('Laser pulse energy (mJ)')
ax[1,1].grid(True)

Fig_K_biases_fname = os.path.join(outpath, 'K_biases_SI.pdf')
plt.savefig(Fig_K_biases_fname, dpi=300)
plt.show()
