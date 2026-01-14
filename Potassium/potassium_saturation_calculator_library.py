# -*- coding: utf-8 -*-

import numpy as np
import netCDF4 as nc
import scipy.optimize as opt
import scipy.interpolate as si
import os

c_light = 2.99792458 * 10**8 #m/s
k_B = 1.38064852 * 10**-23 #J K^-1
h_planck = 6.626*10**-34
N_A = 6.022 * 10**23 #mol^-1
e_elec = 1.60218*10**-19 #C
eps_0 = 8.85419*10**-12 # F/m
m_elec = 9.10938*10**-31 #kg
M_K = 0.0391 #kg mol^-1
m_K = M_K/N_A

lamb0 =  770.10835 * 10**-9 #m, in vacuum
nu0 = c_light / lamb0
A_i = 37.34e6 #Hz
tau_R = 1 / A_i * 1e9 #ns
Delta_nu_n = A_i / (2 * np.pi) #Hz
absorb_coeff = e_elec**2/(4*eps_0 * m_elec * c_light)
sig_D = nu0*np.sqrt(k_B / (m_K * c_light**2))
f_D = .332

#Relative transition strengths
g1_1 = 1
g1_2 = 5
g2_1 = 5
g2_2 = 5

g_jk = np.array([[g1_1, g1_2],[g2_1, g2_2]])

#Relative frequency shift 
nu_11_39 = 253.9e6 #Hz
nu_12_39 = 309.4e6 #Hz
nu_21_39 = -207.8e6 #Hz
nu_22_39 = -152.3e6 #Hz

nu_11_41 = 375.2e6 #Hz
nu_12_41 = 405.7e6 #Hz
nu_21_41 = 121.1e6 #Hz
nu_22_41 = 151.6e6 #Hz

nu_jk = np.array([[[nu_11_39, nu_12_39],[nu_21_39, nu_22_39]],
                  [[nu_11_41, nu_12_41],[nu_21_41, nu_22_41]]])

#Hanle factors - assumed to be 1 here.
q_jk = np.array([[1, 1], [1, 1]]) 

#Relative isotope abundances
f_iso_39 = 0.933
f_iso_41 = 0.067

f_isos = np.array([f_iso_39, f_iso_41])

delta_nu = 1e6
nu_shifts = np.arange(-3*10**9, 3*10**9, delta_nu)
nv = len(nu_shifts)

def convolve(a,b):
    #Calculates the convolution of two arrays and corrects for the shift in
    # index arising from np.convolve
    conv = np.convolve(a,b,'same')*delta_nu
    conv[:-1] = conv[1:]
    return conv

def get_natural_absorption_line(iso, j, k):
   #Returns the scattering cross-section spectrum of the transition between
   # ground-state j and excited state k for isotope 39K (i=0) and 41K (i=1)
   # (natural linewidth only) 
 
    #Need to make sure sufficient resolution is used to resolve absorption
    # line
    
    if nu_shifts[1]-nu_shifts[0] > Delta_nu_n / 5.:
        print('Error: insufficient spectral resolution to resolve' +
              ' absorption line.')
        return
        
    natural_absorption_line = (g_jk[j,k] / np.sum(g_jk[j,:]) * absorb_coeff
                               * f_D / np.pi * Delta_nu_n / 2 / ((nu_shifts -
                              nu_jk[iso,j,k])**2 + (Delta_nu_n/2)**2))
    return natural_absorption_line

def get_combined_absorption_line():
    #Returns the combined absorption line for the complete D_1 line,
    # accounting for the differing relative abundances of the j=1 and j=2
    # ground-states (given by (2j+1)/8, respectively.)
    
    combined_line = np.zeros(nv)
    for iso in range(2):
        for i in range(2):
            for k in range(2):
                j = i+1
                combined_line += (f_isos[iso] * ((2*j+1) / 8) *
                                       get_natural_absorption_line(iso, i, k))
            
    return combined_line

def get_temperature_spectrum(Temp_K): 
    #Returns the Doppler-broadened distribution for a given temperature, i.e.
    # the relative populations of atoms with a corresponding velocity-induced
    # Doppler shift
    sig_D_temp = sig_D * np.sqrt(Temp_K)
    return 1 / np.sqrt(2*np.pi) / sig_D_temp * np.exp(-nu_shifts**2 /
                                                            (2*sig_D_temp**2))

def get_doppler_broadened_spectrum_complete(Temp_K):
    #Returns the Doppler-broadened scattering cross-section spectrum of the
    #complete D_1 line
    
    temp_spectrum = get_temperature_spectrum(Temp_K)
    combined_spectrum = get_combined_absorption_line()
    return convolve(combined_spectrum, temp_spectrum)

def g_L_lorentzian(nu_L, Delta_nu_L):
    #Returns a laser profile with a Lorentzian profile
    return 2 / np.pi / Delta_nu_L * (Delta_nu_L/2)**2 / ((nu_shifts - nu_L)**2
                                                          + (Delta_nu_L/2)**2)

def g_L_gauss(nu_L, Delta_nu_L):
    #Returns a laser profile with a Gaussian profile
    sigma_L = Delta_nu_L / 2.355
    return 1 / sigma_L / np.sqrt(2 * np.pi) * np.exp(-(nu_shifts - nu_L)**2 / 
                                                             (2 * sigma_L**2))

def get_laser_pulseshape(nu_L, Delta_nu_L, lineshape):
    #Returns a laser profile with the given profile
    if lineshape == 'gauss':
        return g_L_gauss(nu_L, Delta_nu_L)
    elif lineshape == 'lorentzian':
        return g_L_lorentzian(nu_L, Delta_nu_L)
    else:
        print('Invalid lineshape: ' + lineshape)
        return None

def get_effective_absorption_lines(nu_L=0, Delta_nu_L = 100*10**6,
                                   lineshape='gauss'):
    #Returns the effective absorption spectrum, accounting for laser lineshape

    #Check that sufficient spectral resolution is used to resolve the
    # laser line
                                       
    if delta_nu > Delta_nu_L / 5.:
        print('Error: insufficient spectral resolution to resolve the' + 
              ' laser line.')
        return 
      
    L_jk = np.zeros((2, 2, 2, nv))
    laser_spectrum = get_laser_pulseshape(nu_L, Delta_nu_L, lineshape)
    
    for iso in range(2):
        for j in range(2):
            for k in range(2):
                alpha_jk = (absorb_coeff * g_jk[j,k] / np.sum(g_jk[j,:])
                            * f_D / np.pi * Delta_nu_n / 2 
                            / ((nu_jk[iso,j,k] + nu_shifts)**2 
                            + (Delta_nu_n/2)**2))
                L_jk[iso,j,k,:] = convolve(laser_spectrum, alpha_jk)

    return L_jk

def get_total_scattering_cross_section_spectrum(Temp_K, Delta_nu_L=100e6,
                                                lineshape='gauss'):
    #Returns the total effective Doppler-broadened scattering cross-section
    # spectrum, accounting for the laser lineshape
                                                  
    temp_spectrum = get_doppler_broadened_spectrum_complete(Temp_K)
    laser_spectrum = get_laser_pulseshape(0, Delta_nu_L, lineshape)
    sigma_tot = convolve(laser_spectrum, temp_spectrum)
    return sigma_tot

def get_total_scattering_cross_section(Temp_K, nu_L=0, Delta_nu_L=100e6,
                                       lineshape='gauss'):
    #Returns the total effective Doppler-broadened scattering cross-section
    # for a given laser frequency
    
    sigma_tot = get_total_scattering_cross_section_spectrum(Temp_K,
                                                            Delta_nu_L,
                                                            lineshape)
    f_sigma = si.interp1d(nu_shifts, sigma_tot)
    return f_sigma(nu_L)


def fit_wind_and_temp(params, nu_Ls, ydata, Delta_nu_L=100e6,
                                 lineshape='gauss'):
    #Fit function for a multi-frequency measurement of the scattering
    # cross-section spectrum, in terms of an amplitude (params[0]),
    # temperature (params[1]) and line-of-sight wind velocity (params[2])
    
    fit = params[0]*get_total_scattering_cross_section(params[1],
                nu_Ls-params[2]/lamb0, Delta_nu_L, lineshape)
    return fit - ydata
  
def plot_fit(params, nu_Ls, Delta_nu_L=100e6, lineshape='gauss'):
    fit = params[0]*get_total_scattering_cross_section(params[1],
                                                       nu_Ls-params[2]/lamb0,
                                                       Delta_nu_L, lineshape)
    return fit

def N_L_from_pulse_energy(E, nu=nu0): 
    #Returns the number of photons per pulse for a pulse energy in mJ
  
    return E/(h_planck*nu) / 1000

def N_t_laser(nt, delta_t, t_L, N_L):
    #Returns the number of photons emitted per time interval, assuming the 
    # laser pulse shape approximation given by VDG
    
    t = np.arange(delta_t/2, nt*delta_t + delta_t/2, delta_t)
    a = 3.4 / t_L 
    return N_L * t**2 * np.exp(-a*t) * a**3 / 2 * delta_t

def get_saturation_megie(z, alpha_L, t_L, sigma_eff, N_L, T_atm):
    #Returns the expected degree of saturation, according to the Megie
    # approach
    
    Omega = np.pi / 4 * alpha_L**2    
    t_s = (z**2 * Omega * t_L) / (2 * sigma_eff * N_L * T_atm)
    return 1 - 1 / (1 + tau_R/t_s) * (1 - tau_R/t_L * tau_R / (t_s + tau_R)
                    * (np.exp(-(t_L / tau_R) * (1 + tau_R / t_s))-1))

def get_saturation(nu_L, Delta_nu_L, N_L, z, alpha_L, T_atm, t_L=10, nt=50, 
                   delta_t=1, Temp_K=200, lineshape='gauss', ratio=False):
    #Returns the expected degree of saturation, according to the VDG approach    
    
    N = N_t_laser(nt, delta_t, t_L, N_L) 
    Omega = np.pi / 4 * alpha_L**2    
    temp_spectrum = get_temperature_spectrum(Temp_K)

    
    #Find the index for which 99.99% of photons have been accounted for, in
    # order to abridge the calculation (the DES can be solved analytically
    # for time steps where the number of photons is approximately zero)
    
    try:
        nt = np.where(np.cumsum(N) > 0.9999 * N_L)[0][0] + 2
    except IndexError:
        print('No abridgement possible. Warning:' +  
              'only {0:.2f} percent of photons accounted for.'.format(
                       np.sum(N)/N_L * 100))    
   
    n = np.zeros((2, 2, nv, nt))
    n_e = np.zeros((2, 2, nv, nt))
    
    n[:,0,:,0] = np.ones(nv) * 3 / 8.
    n[:,1,:,0] = np.ones(nv) * 5 / 8.

    n_e2 = np.zeros((2, 2, nv, nt))

    P_s = 0
    P_ns = 0   
    
    L_jk = get_effective_absorption_lines(nu_L, Delta_nu_L, lineshape)
    
    for i in range(nt - 1):
        number_of_photons = T_atm * N[i] / z**2 / Omega
        f_jk = L_jk * number_of_photons     
        
        if np.max(f_jk) > 0.25:
            print('Error: delta_t must be reduced.' + 
                  ' max f_jk = {0:.3f} '.format(np.max(f_jk)))
            return
        n[:,0,:,i+1] = n[:,0,:,i] + ((1 / 6 * n_e[:,0,:,i]
                                      + 1 / 2 * n_e[:,1,:,i] )
                                      / tau_R  * delta_t
                                      - (n[:,0,:,i] - n_e[:,0,:,i])
                                      * f_jk[:,0,0,:] 
                                      - (n[:,0,:,i] - 3 / 5 * n_e[:,1,:,i])
                                      * f_jk[:,0,1,:])
        n[:,1,:,i+1] = n[:,1,:,i] + ((5 / 6 * n_e[:,0,:,i]
                                      + 1 / 2 * n_e[:,1,:,i])
                                      / tau_R  * delta_t
                                      - (n[:,1,:,i] - 5 / 3 * n_e[:,0,:,i])
                                      * f_jk[:,1,0,:] 
                                      - (n[:,1,:,i] -  n_e[:,1,:,i])
                                      * f_jk[:,1,1,:]) 
        n_e[:,0,:,i+1] = n_e[:,0,:,i] + (-n_e[:,0,:,i] / tau_R * delta_t
                                         + (n[:,0,:,i] - n_e[:,0,:,i])
                                         * f_jk[:,0,0,:]
                                         + (n[:,1,:,i] - 5 / 3 * n_e[:,0,:,i])
                                         * f_jk[:,1,0,:])
        n_e[:,1,:,i+1] = n_e[:,1,:,i] + (-n_e[:,1,:,i] / tau_R * delta_t
                                         + (n[:,0,:,i] - 3 / 5 * n_e[:,1,:,i])
                                         * f_jk[:,0,1,:]
                                         + (n[:,1,:,i] -  n_e[:,1,:,i])
                                         * f_jk[:,1,1,:])

        n_e2[:,0,:,i+1] = n_e2[:,0,:,i] + (-n_e2[:,0,:,i] / tau_R * delta_t
                                    + 0.375 * f_jk[:,0,0,:]
                                    + 0.625 * f_jk[:,1,0,:]) 
        n_e2[:,1,:,i+1] = n_e2[:,1,:,i] + (-n_e2[:,1,:,i] / tau_R * delta_t
                                    + 0.375 * f_jk[:,0,1,:]
                                    + 0.625 * f_jk[:,1,1,:])
        for iso in range(2):
            for j in range(2):
                P_s += f_isos[iso] * (np.sum(n_e[iso,j,:,i] *
                             (g_jk[0,j]/np.sum(g_jk[:,j]) * q_jk[0,j]
                             + g_jk[1,j]/np.sum(g_jk[:,j]) * q_jk[1,j])
                             * temp_spectrum)
                             / tau_R * delta_t / np.sum(temp_spectrum))
                P_ns += f_isos[iso] * (np.sum(n_e2[iso,j,:,i] *
                              (g_jk[0,j]/np.sum(g_jk[:,j]) * q_jk[0,j]
                              + g_jk[1,j]/np.sum(g_jk[:,j]) * q_jk[1,j])
                              * temp_spectrum)
                              / tau_R * delta_t / np.sum(temp_spectrum))
            
    for iso in range(2):
        for j in range(2):
            P_s += f_isos[iso] * np.sum(n_e[iso,j,:,i+1] *
                         (g_jk[0,j]/np.sum(g_jk[:,j]) * q_jk[0,j]
                         + g_jk[1,j]/np.sum(g_jk[:,j]) * q_jk[1,j])
                         * temp_spectrum) / np.sum(temp_spectrum)
            P_ns += f_isos[iso] * np.sum(n_e2[iso,j,:,i+1] *
                          (g_jk[0,j]/np.sum(g_jk[:,j]) * q_jk[0,j]
                          + g_jk[1,j]/np.sum(g_jk[:,j]) * q_jk[1,j])
                          * temp_spectrum) / np.sum(temp_spectrum)     
            
    #If ratio == True, the degree of saturation is returned. If 
    # ratio != True, the total number of emitted photons in the case with 
    # saturation and without saturation are returned individually
    
    if ratio:
        return 1 - P_s / P_ns
    else:
        return P_s, P_ns

def gauss_1D(alpha_L, r):
    #Returns a 1D Gaussian profile
    return np.exp(-4*r**2/alpha_L**2)

def get_saturation_beam(nu_L, Delta_nu_L, N_L, z, T_atm, alpha_L, alpha_T,
                        t_L=10, nt=50, delta_t=1, delta_r = 5 * 10**-6,
                        Temp_K=200, lineshape='gauss', ratio_beam=True):
    #Returns the expected degree of saturation, according to the VDG approach,
    # averaged over a Gaussian beam profile over the field-of-view of the
    # telescope.
    
    #In order to avoid coverage issues, the parameter delta_r is adjusted so
    # that the field-of-view of the instrument is covered by an integer number
    # of equal-width bins.
    
    r_max = alpha_T/2.
    n_r = int(r_max / delta_r)+1
    delta_r_adjusted = r_max / n_r
    
    Omega = np.pi / 4 * alpha_L**2
    
    r = np.arange(delta_r_adjusted/2, r_max, delta_r_adjusted)

    sats = np.zeros((2,len(r)))
    beam = N_L * gauss_1D(alpha_L, r)
    
    for i in range(len(r)):
        sats[:,i] = get_saturation(nu_L, Delta_nu_L, beam[i], z, alpha_L,
                                   T_atm, t_L, nt, delta_t, Temp_K, lineshape,
                                   ratio=False)
    
    #If ratio_beam == True, the degree of saturation is returned. If 
    # ratio_beam != True, the total number of emitted photons in the 
    # case with saturation and without saturation are returned individually
    
    if ratio_beam:
        return 1 - np.sum(sats[0,:] * r) / np.sum(sats[1,:] * r)
    else:
        return np.sum(sats[0,:] * r)/Omega, np.sum(sats[1,:] * r)/Omega
    
def get_wind_and_temp_errors(Temp_K, nu_Ls, Delta_nu_L, N_L, z, T_atm,
                             alpha_L, alpha_T, t_L=10, nt=50, delta_t=1,
                             delta_r=1e-5, lineshape='gauss'):
    #Determines the retrieved spectrum, with and without saturation, at the
    # given measurement frequencies. Fits a spectrum based on the measurements
    # in each case, and returns the fit parameters (amplitude, temperature,
    # and LOS wind speed)
    
    Ps = np.zeros((2,len(nu_Ls)))
    norm = get_total_scattering_cross_section(Temp_K, 0)
    
    for i in range(len(nu_Ls)):
        nu_L = nu_Ls[i]
        Ps[:,i] = get_saturation_beam(nu_L, Delta_nu_L, N_L, z, T_atm,
                                      alpha_L, alpha_T, t_L, nt, delta_t,
                                      delta_r, Temp_K, lineshape, False)
    
    p = np.array([np.max(Ps)/norm, Temp_K, 1])
    res_sat = opt.leastsq(fit_wind_and_temp, p,
                          args=(nu_Ls, Ps[0,:], Delta_nu_L, lineshape),
                          full_output=1)
    res_no_sat = opt.leastsq(fit_wind_and_temp, p,
                             args=(nu_Ls, Ps[1,:], Delta_nu_L, lineshape),
                             full_output=1)
    
    return res_sat, res_no_sat, Ps
  
def get_lidar_residuals(lineshape, Delta_nu_L = 20e6):
    #Data analysis for the potassium lidar data. The data are filtered, 
    # background-subtracted, and normalized, then binned in time and
    # altitude. A three-parameter fit is made to the resulting spectrum
    # of the scattering cross-section, and the residuals of that fit are
    # returned, along with the average temperature over the potassium 
    # layer.

    obs_path = os.path.join(os.path.dirname(os.getcwd()), 'K-Lidar Data')
    fnames = os.listdir(obs_path)
  
    Res_array = []
    Temps = np.zeros(5)
    
    for i in range(5):
        fname = os.path.join(obs_path, fnames[n])
        df = nc.Dataset(fname)

        counts = np.array(df.variables['counts'][:,:,:])
        
        mask1 = np.where(np.sum(counts[:,200:250,:], axis=(1,2)) > 0)[0]
        metric = (np.sum(counts[mask1,200:250,:], axis=(1,2)) /
                              np.sum(counts[mask1,625:,:], axis=(1,2)))
        mask2 = np.where(metric > np.median(metric)/3.)[0]
        mask = mask1[mask2]
        
        counts = counts[mask,:,:]
        wavelengths = np.array(df.variables['wavelength'][:])[::-1] + 0.09
        shots = np.array(df.variables['shots'][mask])
    
        bg = np.mean(counts[:,625:,:], axis=1) / shots
        nu_Ls = -c_light / lamb0**2 * wavelengths*1e-12

        combined_kalium_profiles = []
        for j in range(15):
            profile = np.sum((np.sum(counts[:,425 + 5*j:430 + 5*j,:], axis=1)
                          / shots  - 5*bg),axis=0)
            combined_kalium_profiles.append(profile)
    
        norm = get_total_scattering_cross_section(175, 0)
        obs_fit_combined = []
        dens = []
        temps = []
        winds = []

        for profile in combined_kalium_profiles:
            p = np.array([np.max(profile)/norm, 175, 0.1])
            obs_fit_combined.append(opt.leastsq(fit_wind_and_temp, p,
                                         args=(nu_Ls, profile, Delta_nu_L,
                                               lineshape), full_output=1)[0]) 
            dens.append(obs_fit_combined[-1][0])
            temps.append(obs_fit_combined[-1][1])
            winds.append(obs_fit_combined[-1][2])
    
        resids_combined = []
        
        for j in range(len(combined_kalium_profiles)):
            resids_combined.append(combined_kalium_profiles[j] - 
                                   plot_fit(obs_fit_combined[j], nu_Ls,
                                            Delta_nu_L, lineshape))
    
        Res_array.append(np.array(resids_combined) 
                                      / np.array(combined_kalium_profiles))
        Temps[i] = np.mean(temps)        
    
    return Res_array, Temps
  
  
def get_model_residuals(nu_Ls, Delta_nu_L, N_L, z, alpha_L, alpha_T, T_atm,
                        t_L, nt, delta_t, delta_r, Temp, lineshape):
    #Simulates a multi-wavelength measurement, performs a three-parameter fit
    # to the simulated spectrum, and computes the resulting residuals.
  
    sat_spectrum = np.zeros(len(nu_Ls))

    for i in range(len(nu_Ls)):
        sat_spectrum[i], _ = get_saturation_beam(nu_Ls[i], Delta_nu_L, N_L, z,
                                                 T_atm, alpha_L, alpha_T, t_L,
                                                 nt, delta_t, delta_r, Temp,
                                                 lineshape, ratio_beam=False)
        
    norm = get_total_scattering_cross_section(Temp, 0)
    p = np.array([np.max(sat_spectrum)/norm, Temp, 0.1])
    model_fit = opt.leastsq(fit_wind_and_temp, p,
                            args=(nu_Ls, sat_spectrum, Delta_nu_L, lineshape),
                            full_output=1)[0]

    model_resid = (sat_spectrum - plot_fit(model_fit, nu_Ls, Delta_nu_L,
                                           lineshape)) / sat_spectrum
    
    return model_resid 


