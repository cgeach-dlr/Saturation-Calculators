# -*- coding: utf-8 -*-

import sys
sys.path.append("")
import potassium_saturation_calculator_library as k_lib
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

font = {'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

#Designate an output location for figure data and plots
outpath = os.path.join(os.path.dirname(os.getcwd()), 'Output')

#Get lidar count profiles
lidar_data = k_lib.get_lidar_count_profiles()

fig, ax = plt.subplots(1,2,figsize=(12,5))
ax[0].plot(lidar_data[0][0,:,8][120:], np.arange(750)[120:]/5.,
           label='{0:.2f} pm'.format(lidar_data[2][8]), alpha=0.75)
ax[0].plot(lidar_data[0][0,:,0][120:], np.arange(750)[120:]/5.,
           label='{0:.2f} pm'.format(lidar_data[2][0]), alpha=0.75)
ax[0].set_xlabel('Counts / shot')
ax[0].set_ylabel('Altitude (km)')
ax[0].set_xscale('log')
ax[0].legend()

ax[1].bar(lidar_data[2], lidar_data[1][0], 0.1)
ax[1].set_xlabel('Wavelength (pm)')
ax[1].set_ylabel('Number of shots')

fig.tight_layout()
Fig_K_raw_data_fname = os.path.join(outpath, 'K_raw_data.pdf')
plt.savefig(Fig_K_raw_data_fname, dpi=300, bbox_inches = 'tight')
plt.show()

Data_K_profile = np.hstack((np.arange(750)[120:]/5., 
                            lidar_data[0][0,:,8][120:],
                            lidar_data[0][0,:,0][120:]))
Data_K_profile_fname = os.path.join(outpath, 'K_profile.txt')
np.savetxt(Data_K_profile_fname, Data_K_profile, delimiter=',')

Data_K_shots = np.hstack((lidar_data[2], lidar_data[1][0]))
Data_K_shots_fname = os.path.join(outpath, 'K_shots.txt')
np.savetxt(Data_K_shots_fname, Data_K_shots, delimiter=',')


#Calculate experimental residuals, noise, and temperatures
R_gauss_data, N_gauss, T_gauss = k_lib.get_lidar_res('gauss')
R_lorentz_data, N_lorentz, T_lorentz = k_lib.get_lidar_res('lorentzian')

#Calculate model residuals
lambda_Ls = np.arange(1.55, -1.52, -0.18)*1e-12 #m
nu_Ls = -k_lib.c_light / k_lib.lamb0**2 * lambda_Ls

E_pulse = 50 #mJ
t_L = 275 #ns
T_atm = 0.7
z = 92.5e3 #m
alpha_L = 270e-6 #rad
alpha_T = 186e-6 #rad

nt = 2000
delta_t = 1 #ns
delta_r = 186e-6 #rad
N_L = k_lib.N_L_from_pulse_energy(E_pulse)
Delta_nu_L = 20e6 #Hz

R_gauss_model = []
R_lorentz_model = []

for i in range(5):
    R_gauss_model.append(k_lib.get_model_res(nu_Ls, Delta_nu_L, N_L, z, 
                                        alpha_L, alpha_T, T_atm, t_L,
                                        nt, delta_t, delta_r, T_gauss[i],
                                        'gauss', np.mean(N_gauss[i], axis=0)))
    R_lorentz_model.append(k_lib.get_model_res(nu_Ls, Delta_nu_L, N_L, z, 
                                               alpha_L, alpha_T, T_atm, t_L,
                                               nt, delta_t, delta_r, 
                                               T_lorentz[i], 'lorentzian',
                                               np.mean(N_lorentz[i], axis=0)))
    
dates = ['25-26 January 2010', '26 January 2010 (evening)',
         '27-28 January 2011', '28-29 January 2011', '24-25 February 2011']

lineshape_names = ['Gauss', 'Lorentzian']

fig = plt.figure(figsize=(12,7))
for i in range(15):
    plt.plot(lambda_Ls * 1e12, 100*R_lorentz_data[4][i])
    
plt.xlabel('Wavelength offset (pm)')
plt.ylabel('Relative residuals (%)') 
plt.axhline(0, c='k')

Fig_K_measurements_fname = os.path.join(outpath, 'K_measurements.pdf')
plt.savefig(Fig_K_measurements_fname, dpi=300)
plt.show()

Data_K_measurements = np.hstack((lambda_Ls[:, np.newaxis] * 1e12,
                                 100*R_lorentz_data[2].T))
Data_K_measurements_fname = os.path.join(outpath, 'K_measurements.txt')
np.savetxt(Data_K_measurements_fname, Data_K_measurements, delimiter=',')

figure_parts = np.array([['a','b'],
                         ['c','d'],
                         ['e','f']])

ns = np.array([2,3,4])    
fig, ax = plt.subplots(3,2, figsize=(16,20))
for i in range(3):
    for j in range(2):
        if j == 0:
            R_data = R_gauss_data
            R_model = R_gauss_model
        else:
            R_data = R_lorentz_data
            R_model = R_lorentz_model
        
        n = ns[i]
        
        data_res = R_data[n]
        model_res = R_model[n]
        
        ax[i,j].set_title(dates[n] + ' -- ' + lineshape_names[j] + ' profile')
        ax[i,j].plot(lambda_Ls * 1e12, 100*np.mean(data_res, axis=0), 'k',
                     label='Observed residuals')
        ax[i,j].plot(lambda_Ls * 1e12, 100*(np.mean(data_res, axis=0) +
                                             np.std(data_res, axis=0)), 'k--')
        ax[i,j].plot(lambda_Ls * 1e12, 100*(np.mean(data_res, axis=0) -
                                             np.std(data_res, axis=0)), 'k--')
        ax[i,j].plot(lambda_Ls * 1e12, 100*model_res, 'r',
                     label='Model residuals')
        ax[i,j].set_ylim(-12,15.5)
        ax[i,j].axhline(0)
        
        
        Data_K_comp = np.vstack((lambda_Ls * 1e12, 
                                 100*np.mean(data_res, axis=0),
                                 100*(np.mean(data_res, axis=0)
                                 + np.std(data_res, axis=0)),
                                 100*(np.mean(data_res, axis=0)
                                 - np.std(data_res, axis=0))))
        Data_K_comp_fname = os.path.join(outpath, 'K_comparison_'
                                       + figure_parts[i,j] + '.txt')
        np.savetxt(Data_K_comp_fname, Data_K_comp.T, delimiter=',')
        
        
        if i == 2:        
            ax[i,j].set_xlabel('Wavelength offset (pm)')
        else:
            ax[i,j].set_xticks([])
    ax[i,0].set_ylabel('Relative residuals (%)') 
    ax[i,1].set_yticks([])
ax[0,0].legend()
      
fig.tight_layout()
Fig_K_comp_fname = os.path.join(outpath, 'K_comparison.pdf')
plt.savefig(Fig_K_comp_fname, dpi=300)
plt.show()


ns = np.array([0,1])
fig, ax = plt.subplots(2,2, figsize=(16,13))
for i in range(2):
    for j in range(2):    
        if j == 0:
            R_data = R_gauss_data
            R_model = R_gauss_model
        else:
            R_data = R_lorentz_data
            R_model = R_lorentz_model
        
        n = ns[i]
            
        data_res = R_data[n]
        model_res = R_model[n]
        
        ax[i,j].set_title(dates[n] + ' -- ' + lineshape_names[j] + ' profile')
        ax[i,j].plot(lambda_Ls * 1e12, 100*np.mean(data_res, axis=0), 'k',
                     label='Observed residuals')
        ax[i,j].plot(lambda_Ls * 1e12, 100*(np.mean(data_res, axis=0) +
                                            np.std(data_res, axis=0)), 'k--')
        ax[i,j].plot(lambda_Ls * 1e12, 100*(np.mean(data_res, axis=0) - 
                                            np.std(data_res, axis=0)), 'k--')
        ax[i,j].plot(lambda_Ls * 1e12, 100*model_res, 'r',
                     label='Model residuals')
        ax[i,j].set_ylim(-12,15.5)
        ax[i,j].axhline(0)
        
        
        Data_K_comp2 = np.vstack((lambda_Ls * 1e12, 
                                  100*np.mean(data_res, axis=0),
                                  100*(np.mean(data_res, axis=0)
                                  + np.std(data_res, axis=0)),
                                  100*(np.mean(data_res, axis=0)
                                  - np.std(data_res, axis=0))))
        Data_K_comp2_fname = os.path.join(outpath, 'K_comparison2_' + 
                                               figure_parts[i,j] + '.txt')
        np.savetxt(Data_K_comp2_fname, Data_K_comp2.T, delimiter=',')
        
        
        if i == 1: 
            ax[i,j].set_xlabel('Wavelength offset (pm)')
        else:
            ax[i,j].set_xticks([])
    ax[i,0].set_ylabel('Relative residuals (%)') 
    ax[i,1].set_yticks([])
ax[0,0].legend()
      
fig.tight_layout()
Fig_K_comp2_fname = os.path.join(outpath, 'K_comparison2.pdf')
plt.savefig(Fig_K_comp2_fname, dpi=300)
plt.show()
