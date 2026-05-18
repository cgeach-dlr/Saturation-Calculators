# -*- coding: utf-8 -*-

import sys
sys.path.append("")
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

font = {'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

#Designate an output location for figure data and plots
outpath = os.path.join(os.path.dirname(os.getcwd()), 'Output')

Data_K_profile_fname = os.path.join(outpath, 'Figure Data', 'K_profile.txt')
Data_K_shots_fname = os.path.join(outpath, 'Figure Data', 'K_shots.txt')
Data_K_profile = np.loadtxt(Data_K_profile_fname, delimiter=',').T
Data_K_shots = np.loadtxt(Data_K_shots_fname, delimiter=',').T

altitudes = Data_K_profile[0]
lidar_profile1 = Data_K_profile[1]
lidar_profile2 = Data_K_profile[2]
wavelengths = Data_K_shots[0]
shots = Data_K_shots[1]

fig, ax = plt.subplots(1,2,figsize=(12,5))
ax[0].plot(lidar_profile1, altitudes,
           label='{0:.2f} pm'.format(wavelengths[8]), alpha=0.75)
ax[0].plot(lidar_profile2, altitudes,
           label='{0:.2f} pm'.format(wavelengths[0]), alpha=0.75)
ax[0].set_xlabel('Counts / shot')
ax[0].set_ylabel('Altitude (km)')
ax[0].set_xscale('log')
ax[0].legend()

ax[1].bar(wavelengths, shots, 0.1)
ax[1].set_xlabel('Wavelength (pm)')
ax[1].set_ylabel('Number of shots')

fig.tight_layout()
Fig_K_raw_data_fname = os.path.join(outpath, 'K_raw_data.pdf')
plt.savefig(Fig_K_raw_data_fname, dpi=300, bbox_inches = 'tight')
plt.show()
    
dates1 = ['27-28 January 2011', '28-29 January 2011', '24-25 February 2011']

lineshape_names = ['Gaussian', 'Lorentzian']

Data_K_measurements_fname = os.path.join(outpath, 'Figure Data', 
                                         'K_measurements.txt')
Data_K_measurements = np.loadtxt(Data_K_measurements_fname, delimiter=',').T

fig = plt.figure(figsize=(12,7))
for i in range(15):
    plt.plot(Data_K_measurements[0], Data_K_measurements[1][i])
    
plt.xlabel('Wavelength offset (pm)')
plt.ylabel('Relative residuals (%)') 
plt.axhline(0, c='k')

Fig_K_measurements_fname = os.path.join(outpath, 'K_measurements.pdf')
plt.savefig(Fig_K_measurements_fname, dpi=300)
plt.show()

figure_parts = np.array([['a','b'],
                         ['c','d'],
                         ['e','f']])

fig, ax = plt.subplots(3,2, figsize=(16,20))
for i in range(3):
    for j in range(2):
        Data_K_comp_fname = os.path.join(outpath, 'Figure Data', 
                                         'K_comparison_' + figure_parts[i,j]
                                         + '.txt')
        Data_K_comp = np.loadtxt(Data_K_comp_fname, delimiter=',').T

        ax[i,j].set_title(dates1[i] + ' -- ' + lineshape_names[j] + ' profile')
        ax[i,j].plot(Data_K_comp[0], Data_K_comp[1], 'k',
                     label='Observed residuals')
        ax[i,j].plot(Data_K_comp[0], Data_K_comp[2], 'k--')
        ax[i,j].plot(Data_K_comp[0], Data_K_comp[3], 'k--')
        ax[i,j].plot(Data_K_comp[0], Data_K_comp[4], 'r',
                     label='Model residuals')
        ax[i,j].set_ylim(-12,15.5)
        ax[i,j].axhline(0)

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

dates2 = ['25-26 January 2010', '26 January 2010 (evening)']

fig, ax = plt.subplots(2,2, figsize=(16,13))
for i in range(2):
    for j in range(2):    
        Data_K_comp_fname = os.path.join(outpath, 'Figure Data', 
                                         'K_comparison2_' + figure_parts[i,j]
                                         + '.txt')
        Data_K_comp = np.loadtxt(Data_K_comp_fname, delimiter=',').T
        
        ax[i,j].set_title(dates2[i] + ' -- ' + lineshape_names[j] + ' profile')
        ax[i,j].plot(Data_K_comp[0], Data_K_comp[1], 'k',
                     label='Observed residuals')
        ax[i,j].plot(Data_K_comp[0], Data_K_comp[2], 'k--')
        ax[i,j].plot(Data_K_comp[0], Data_K_comp[3], 'k--')
        ax[i,j].plot(Data_K_comp[0], Data_K_comp[4], 'r',
                     label='Model residuals')
        ax[i,j].set_ylim(-12,15.5)
        ax[i,j].axhline(0)
        
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
