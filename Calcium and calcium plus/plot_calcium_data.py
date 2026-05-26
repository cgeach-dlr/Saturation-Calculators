# -*- coding: utf-8 -*-

import sys
sys.path.append("")
import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib

font = {'weight' : 'normal',
        'size'   : 24}

matplotlib.rc('font', **font)

#Designate an output location for figure data and plots
outpath = os.path.join(os.path.dirname(os.getcwd()), 'Output')

Data_calcium_spectrum_fname1 = os.path.join(outpath, 'Figure Data',
                                     'Ca_and_ca_plus_spectrum1.txt')
Data_calcium_spectrum_fname2 = os.path.join(outpath, 'Figure Data',
                                     'Ca_and_ca_plus_spectrum2.txt')
Data_calcium_spectrum1 = np.loadtxt(Data_calcium_spectrum_fname1,
           delimiter=',').T
Data_calcium_spectrum2 = np.loadtxt(Data_calcium_spectrum_fname2,
           delimiter=',').T
                                    
lambda_Ls = Data_calcium_spectrum1[0]
ca_sats1 = Data_calcium_spectrum1[1]
ca_plus_sats1 = Data_calcium_spectrum1[2]

ca_sats2 = Data_calcium_spectrum2[1]
ca_sats3 = Data_calcium_spectrum2[2]
ca_plus_sats2 = Data_calcium_spectrum2[3]
ca_plus_sats3 = Data_calcium_spectrum2[4]

# Generate plot for Figure 1
plt.figure(figsize=(12,8))

plt.plot(lambda_Ls, ca_sats1, label='Ca')
plt.plot(lambda_Ls, ca_plus_sats1, label='Ca+')
plt.legend()
plt.ylabel('Saturation percent')
plt.xlabel('Laser wavelength offset (pm)')

plt.grid(True)

plt.savefig(os.path.join(outpath, 'Ca_and_ca_plus1.pdf'), dpi=300)

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

plt.savefig(os.path.join(outpath, 'Ca_and_ca_plus3.pdf'), dpi=300)

Data_calcium_biases_fname1 = os.path.join(outpath, 'Figure Data',
                                     'Ca_and_ca_plus_biases1.txt')
Data_calcium_biases_fname2 = os.path.join(outpath, 'Figure Data',
                                     'Ca_and_ca_plus_biases2.txt')
Data_calcium_biases1 = np.loadtxt(Data_calcium_biases_fname1,
           delimiter=',').T
Data_calcium_biases2 = np.loadtxt(Data_calcium_biases_fname2,
           delimiter=',').T

Es = Data_calcium_biases1[0]

ca_T_err1 = Data_calcium_biases1[1]
ca_T_err2 = Data_calcium_biases2[1]
ca_T_err3 = Data_calcium_biases2[2]
ca_plus_T_err1 = Data_calcium_biases1[2]
ca_plus_T_err2 = Data_calcium_biases2[3]
ca_plus_T_err3 = Data_calcium_biases2[4]

# Generate plot for Figure 2       
plt.figure(figsize=(12,8))

plt.plot(Es, ca_T_err1, label='Ca')
plt.plot(Es, ca_plus_T_err1, label='Ca+')
plt.legend()
plt.ylabel('Temperature bias (K)')
plt.xlabel('Pulse energy (mJ)')

plt.grid(True)

plt.savefig(os.path.join(outpath, 'Ca_and_ca_plus2.pdf'), dpi=300)

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

plt.savefig(os.path.join(outpath, 'Ca_and_ca_plus4.pdf'), dpi=300)
