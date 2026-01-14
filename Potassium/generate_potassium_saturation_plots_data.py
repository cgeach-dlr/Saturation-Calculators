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

#Designate an output location for figure data and plots
outpath = ''

Res_gauss_data, Temps_gauss = k_lib.get_lidar_residuals('gauss')
Res_lorentzian_data, Temps_lorentzian = k_lib.get_lidar_residuals('lorentzian')

lambda_Ls = np.arange(1.55, -1.52, -0.18)*1e-12
fig = plt.figure(figsize=(12,7))
for i in range(15):
    plt.plot(lambda_Ls * 1e12, 100*Res_lorentzian_data[2][i])
    
plt.xlabel('Wavelength (pm)')
plt.ylabel('Relative residuals (%)') 
plt.axhline(0, c='k')

plt.savefig(os.path.join(outpath_figs, 'K_measurements.pdf'), dpi=300)
plt.show()

lineshapes = ['lorentzian', 'gauss']

lambda_Ls = np.arange(1.55, -1.52, -0.18)*1e-12 #m
nu_Ls = -c_light / lamb0**2 * lambda_Ls
sats_gauss = np.zeros(len(nu_Ls))
sats_lorentzian = np.zeros(len(nu_Ls))

E_pulse = 50 #mJ
t_L = 275 #ns
T_atm = 0.7
z = 90e3 #m
alpha_L = 270e-6 #rad
alpha_T = 186e-6 #rad

nt = 2000
delta_t = 1 #ns
delta_r = 186e-6 #rad
N_L = k_lib.N_L_from_pulse_energy(E_pulse)
Delta_nu_L = 20e6 #Hz

Res_gauss_model = []
Res_lorentzian_model = []

for i in range(5):
    Res_gauss_model.append(k_lib.get_model_residuals(nu_Ls, Delta_nu_L, N_L, z,
                                                     alpha_L, alpha_T, T_atm,
                                                     t_L, nt, delta_t, delta_r,
                                                     Temps_gauss[i], 'gauss'))
    Res_lorentzian_model.append(k_lib.get_model_residuals(nu_Ls, Delta_nu_L,
                                                          N_L, z, alpha_L,
                                                          alpha_T, T_atm, t_L,
                                                          nt, delta_t, delta_r,
                                                          Temps_lorentzian[i],
                                                          'lorentzian'))
    
names = ['25-27 January 2010', '27-28 January 2011', '24-26 February 2011',
         '14-15 January 2012', '30-31 January 2012']
lineshapes = ['lorentzian', 'gauss']
lineshape_names = ['Lorentzian', 'Gauss']

fig = plt.figure(figsize=(12,7))
for i in range(15):
    plt.plot(lambda_Ls * 1e12, 100*Res_lorentzian_data[2][i])
    
plt.xlabel('Wavelength (pm)')
plt.ylabel('Relative residuals (%)') 
plt.axhline(0, c='k')

plt.savefig(os.path.join(outpath, 'K_measurements.pdf'), dpi=300)
plt.show()

Data_K_measurements = np.hstack((lambda_Ls[:, None] * 1e12,
                                 100*Res_lorentzian_data[2].T))
np.savetxt(os.path.join(outpath, 'K_measurements.txt'),
           Data_K_measurements, delimiter=',')

figure_parts = np.array([['a','b'],
                         ['c','d'],
                         ['e','f']])
    
fig, ax = plt.subplots(3,2, figsize=(16,20))
for i in range(3):
    for j in range(2):
        if j == 0:
            Res_data = Res_gauss_data
            Res_model = Res_gauss_model
        else:
            Res_data = Res_lorentzian_data
            Res_model = Res_lorentzian_model
        
        data_resid = Res_data[i]
        model_resid = Res_model[i]
        
        ax[i,j].set_title(names[i] + ' -- ' + lineshape_names[j] + ' profile')
        ax[i,j].plot(lambda_Ls * 1e12, 100*np.mean(data_resid, axis=0), 'k',
                                                     label='Observed residuals')
        ax[i,j].plot(lambda_Ls * 1e12, 100*(np.mean(data_resid, axis=0) +
                                             np.std(data_resid, axis=0)), 'k--')
        ax[i,j].plot(lambda_Ls * 1e12, 100*(np.mean(data_resid, axis=0) -
                                             np.std(data_resid, axis=0)), 'k--')
        ax[i,j].plot(lambda_Ls * 1e12, 100*model_resid, 'r',
                                                        label='Model residuals')
        ax[i,j].set_ylim(-6,12.5)
        ax[i,j].axhline(0)
        
        Data_K_comparison = np.vstack((lambda_Ls * 1e12,
                                       100*np.mean(data_resid, axis=0),
                100*(np.mean(data_resid, axis=0) + np.std(data_resid, axis=0)),
                100*(np.mean(data_resid, axis=0) - np.std(data_resid, axis=0))))
        np.savetxt(os.path.join(outpath, 'K_comparison_' +
                                       figure_parts[i,j] + '.txt'),
                                       Data_K_comparison.T, delimiter=',')
        
        if i == 2:        
            ax[i,j].set_xlabel('Wavelength (pm)')
        else:
            ax[i,j].set_xticks([])
    ax[i,0].set_ylabel('Relative residuals (%)') 
    ax[i,1].set_yticks([])
ax[0,0].legend()
      
fig.tight_layout()
plt.savefig(os.path.join(outpath, 'K_comparison.pdf'), dpi=300)
plt.show()


fig, ax = plt.subplots(2,2, figsize=(16,13))
for i in range(2):
    for j in range(2):    
        if j == 0:
            Res_data = Res_gauss_data
            Res_model = Res_gauss_model
        else:
            Res_data = Res_lorentzian_data
            Res_model = Res_lorentzian_model
        
        n = i + 3
        data_resid = Res_data[n]
        model_resid = Res_model[n]
        
        ax[i,j].set_title(names[n] + ' -- ' + lineshape_names[j] + ' profile')
        ax[i,j].plot(lambda_Ls * 1e12, 100*np.mean(data_resid, axis=0), 'k',
                                              label='Observed residuals')
        ax[i,j].plot(lambda_Ls * 1e12, 100*(np.mean(data_resid, axis=0) +
                                            np.std(data_resid, axis=0)), 'k--')
        ax[i,j].plot(lambda_Ls * 1e12, 100*(np.mean(data_resid, axis=0) - 
                                            np.std(data_resid, axis=0)), 'k--')
        ax[i,j].plot(lambda_Ls * 1e12, 100*model_resid, 'r',
                                              label='Model residuals')
        ax[i,j].set_ylim(-12,18.5)
        ax[i,j].axhline(0)
        
        Data_K_comparison = np.vstack((lambda_Ls * 1e12,
                                       100*np.mean(data_resid, axis=0),
                100*(np.mean(data_resid, axis=0) + np.std(data_resid, axis=0)),
                100*(np.mean(data_resid, axis=0) - np.std(data_resid, axis=0))))
        np.savetxt(os.path.join(outpath_data, 'K_comparison2_' + 
                                figure_parts[i,j] + '.txt'),
                                Data_K_comparison.T, delimiter=',')
        
        if i == 1:        
            ax[i,j].set_xlabel('Wavelength (pm)')
        else:
            ax[i,j].set_xticks([])
    ax[i,0].set_ylabel('Relative residuals (%)') 
    ax[i,1].set_yticks([])
ax[0,0].legend()
      
fig.tight_layout()
plt.savefig(os.path.join(outpath, 'K_comparison2.pdf'), dpi=300)
plt.show()
