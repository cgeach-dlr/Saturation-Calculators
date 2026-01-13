# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.interpolate as si

c_light = 2.99792458 * 10**8 #m/s
k_B = 1.38064852 * 10**-23 #J K^-1
h_planck = 6.626*10**-34
N_A = 6.022 * 10**23 #mol^-1
e_elec = 1.60218*10**-19 #C
eps_0 = 8.85419*10**-12 # F/m
m_elec = 9.10938*10**-31 #kg
M_Na = 0.02299 #kg mol^-1
m_Na = M_Na/N_A

lamb0 = 589.1583 * 10**-9 #m, in vacuum
nu0 = c_light / lamb0
A_i = 61.6e6 #Hz
tau_R = 1 / A_i * 1e9 #ns
Delta_nu_n = A_i / (2 * np.pi) #Hz
absorb_coeff = e_elec**2/(4*eps_0 * m_elec * c_light)
sig_D = nu0*np.sqrt(k_B / (m_Na * c_light**2))
f_D = 0.641

#Relative transition strengths
g1_0 = 2
g1_1 = 5
g1_2 = 5
g2_1 = 1
g2_2 = 5
g2_3 = 14

#Relative frequency shifts
nu_10 = 1.0408 * 10**9 #Hz 
nu_11 = 1.0566 * 10**9 #Hz
nu_12 = 1.0911 * 10**9 #Hz
nu_21 = -0.715 * 10**9 #Hz
nu_22 = -0.6806 * 10**9 #Hz
nu_23 = -0.6216 * 10**9 #Hz

g_jk = np.array([[g1_0, g1_1, g1_2],[g2_1, g2_2, g2_3]])
nu_jk = np.array([[nu_10, nu_11, nu_12], [nu_21, nu_22, nu_23]])

#Hanle factors - assumed to be 1 here.
q_jk = np.array([[1, 1, 1], [1, 1, 1]]) 

delta_nu = 1e6
nu_shifts = np.arange(-3*10**9, 3*10**9, delta_nu)
nv = len(nu_shifts)

def convolve(a,b):
    #Calculates the convolution of two arrays and corrects for the shift in 
    # index arising from np.convolve
    
    conv = np.convolve(a,b,'same')*delta_nu
    conv[:-1] = conv[1:]
    return conv

def get_natural_absorption_line(j, k):
   #Returns the scattering cross-section spectrum of the transition
   # between ground-state j and excited state k (natural linewidth only)
 
    #Check that sufficient spectral resolution is used to resolve the
    # absorption line
    
    if nu_shifts[1]-nu_shifts[0] > Delta_nu_n / 5.:
        print('Error: insufficient spectral resolution to resolve absorption' + 
              'line.')
        return
        
    natural_absorption_line = (g_jk[j,k] / np.sum(g_jk[j,:]) * absorb_coeff *
                               f_D / np.pi * Delta_nu_n / 2 / ((nu_shifts - 
                               nu_jk[j,k])**2 + (Delta_nu_n/2)**2))
    return natural_absorption_line

def get_combined_absorption_line():
    #Returns the combined absorption line for the complete D_2 line, accounting
    #for the differing relative abundances of the j=1 and j=2 ground-states 
    # (given by (2j+1)/8, respectively.)
    
    combined_line = np.zeros(len(nu_shifts))
    for i in range(2):
        for k in range(3):
            j = i+1
            combined_line += ((2*j + 1) / 8) * get_natural_absorption_line(i, k)
            
    return combined_line


def get_temperature_spectrum(Temp_Na): 
    #Returns the Doppler-broadened distribution for a given temperature, i.e. 
    # the relative populations of atoms with a corresponding velocity-induced 
    # Doppler shift
    
    sig_D_temp = sig_D * np.sqrt(Temp_Na)
    return 1 / np.sqrt(2*np.pi) / sig_D_temp * np.exp(-nu_shifts**2
                                                      / (2*sig_D_temp**2))

def get_doppler_broadened_spectrum_complete(Temp_Na):
    #Returns the Doppler-broadened scattering cross-section spectrum of the
    # complete D_2 line
    
    temp_spectrum = get_temperature_spectrum(Temp_Na)
    combined_spectrum = get_combined_absorption_line()
    return convolve(combined_spectrum, temp_spectrum)

def g_L_lorentzian(nu_L, Delta_nu_L):
    #Returns a laser profile with a Lorentzian profile
    
    return 2 / np.pi / Delta_nu_L * (Delta_nu_L/2)**2 / ((nu_shifts - nu_L)**2
                                                         + (Delta_nu_L/2)**2)

def g_L_gauss(nu_L, Delta_nu_L):
    #Returns a laser profile with a Gaussian profile
    
    sigma_L = Delta_nu_L / 2.355
    return 1 / sigma_L / np.sqrt(2 * np.pi) * np.exp(-(nu_shifts - nu_L)**2
                                                     / (2 * sigma_L**2))

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
    #Returns the effective absorption spectra, accounting for laser lineshape

    #Check that sufficient spectral resolution is used to resolve the
    # laser line
                                       
    if delta_nu > Delta_nu_L / 5.:
        print('Error: insufficient spectral resolution to resolve the' + 
              ' laser line.')
        return
                                       
    L_jk = np.zeros((2, 3, len(nu_shifts)))
    laser_spectrum = get_laser_pulseshape(nu_L, Delta_nu_L, lineshape)
    
    for j in range(2):
        for k in range(3):
            alpha_jk = (absorb_coeff * g_jk[j,k] / np.sum(g_jk[j,:]) * (f_D 
                        / np.pi * Delta_nu_n / 2 / ((nu_shifts + nu_jk[j,k])**2
                                                     + (Delta_nu_n/2)**2)))
            L_jk[j,k] = convolve(laser_spectrum, alpha_jk)
    return L_jk

def get_total_scattering_cross_section_spectrum(Temp_Na, Delta_nu_L=100e6,
                                                lineshape='gauss'):
    #Returns the total effective Doppler-broadened scattering cross-section
    # spectrum, accounting for the laser lineshape
    
    temp_spectrum = get_doppler_broadened_spectrum_complete(Temp_Na)
    laser_spectrum = get_laser_pulseshape(0, Delta_nu_L, lineshape)
    sigma_tot = convolve(laser_spectrum, temp_spectrum)
    return sigma_tot

def get_total_scattering_cross_section(Temp_Na, nu_L=0, Delta_nu_L=100e6,
                                       lineshape='gauss'):
    #Returns the total effective Doppler-broadened scattering cross-section for
    # a given laser frequency
    
    sigma_tot = get_total_scattering_cross_section_spectrum(Temp_Na, Delta_nu_L,
                                                                      lineshape)
    f_sigma = si.interp1d(nu_shifts, sigma_tot)
    return f_sigma(nu_L)


def fit_wind_and_temp(params, nu_Ls, ydata, Delta_nu_L=100e6,
                      lineshape='gauss'):
    #Fit function for a multi-frequency measurement of the scattering 
    # cross-section spectrum, in terms of an amplitude (params[0]), temperature
    # (params[1]) and line-of-sight wind velocity (params[2])
    
    fit = params[0]*get_total_scattering_cross_section(params[1], 
                                 nu_Ls - params[2]/lamb0, Delta_nu_L, lineshape)
    return fit - ydata

def N_L_from_pulse_energy(E, nu=nu0): 
    #Returns the number of photons per pulse for a pulse energy in mJ
    return E/(h_planck*nu) / 1000

def N_t_laser(nt, delta_t, t_L, N_L):
    #Returns the number of photons emitted per time interval, assuming the laser
    # pulse shape approximation given by VDG
    
    t = np.arange(delta_t/2, nt*delta_t + delta_t/2, delta_t)
    a = 3.4 / t_L 
    return N_L * t**2 * np.exp(-a*t) * a**3 / 2 * delta_t

def get_saturation_megie(z, alpha_L, t_L, sigma_eff, N_L, T_atm):
    #Returns the expected degree of saturation, according to the Megie approach
    
    Omega = np.pi / 4 * alpha_L**2    
    t_s = (z**2 * Omega * t_L) / (2 * sigma_eff * N_L * T_atm)
    return 1 - 1 / (1 + tau_R/t_s) * (1 - tau_R/t_L * tau_R / (t_s + tau_R)
                    * (np.exp(-(t_L / tau_R) * (1 + tau_R / t_s))-1))

def get_saturation(nu_L, Delta_nu_L, N_L, z, alpha_L, T_atm, t_L=10, nt=50,
                   delta_t=1, Temp_Na=200, lineshape='gauss', ratio=False):
    #Returns the expected degree of saturation, according to the VDG approach    
    
    N = N_t_laser(nt, delta_t, t_L, N_L) 
    Omega = np.pi / 4 * alpha_L**2    
    temp_spectrum = get_temperature_spectrum(Temp_Na)
    
    #Find the index for which 99.999% of photons have been accounted for, in 
    # order to abridge the calculation (the DES can be solved analytically for
    # time steps where the number of photons is approximately zero)
                       
    try:
        nt = np.where(np.cumsum(N) > 0.99999 * N_L)[0][0] + 2
    except IndexError:
        print('No abridgement possible. Warning:' +  
              'only {0:.2f} percent of photons accounted for.'.format(
                                                        np.sum(N)/N_L * 100))    
   
    n = np.zeros((2, nv, nt))
    n_e = np.zeros((4, nv, nt))
    
    n[0,:,0] = np.ones(nv) * 3 / 8.
    n[1,:,0] = np.ones(nv) * 5 / 8.

    n_e2 = np.zeros((4, nv, nt))

    P_s = 0
    P_ns = 0
             
    
    L_jk = get_effective_absorption_lines(nu_L, Delta_nu_L, lineshape)
    
    for i in range(nt - 1):
        number_of_photons = T_atm * N[i] / z**2 / Omega
        f_jk = L_jk * number_of_photons     
        
        if np.max(f_jk) > 0.1:
            print('Error: delta_t must be reduced. max f_jk = {0:.3f} '.format(
                                                                 np.max(f_jk)))
            return
        
        n[0,:,i+1] = n[0,:,i] + ((n_e[0,:,i] + 5 / 6 * n_e[1,:,i] +
                                  1 / 2 * n_e[2,:,i]) / tau_R * delta_t
                                - (n[0,:,i] - 3 * n_e[0,:,i]) * f_jk[0,0,:] 
                                - (n[0,:,i] -  n_e[1,:,i]) * f_jk[0,1,:] 
                                - (n[0,:,i] - 3 / 5 * n_e[2,:,i]) * f_jk[0,2,:]) 
        n[1,:,i+1] = n[1,:,i] + ((1 / 6 * n_e[1,:,i] + 1 / 2 * n_e[2,:,i] +
                                  n_e[3,:,i]) / tau_R * delta_t 
                                - (n[1,:,i] - 5 / 3 * n_e[1,:,i]) * f_jk[1,0,:] 
                                - (n[1,:,i] -  n_e[2,:,i]) * f_jk[1,1,:] 
                                - (n[1,:,i] - 5 / 7 * n_e[3,:,i]) * f_jk[1,2,:])
        n_e[0,:,i+1] = n_e[0,:,i] + (-n_e[0,:,i] / tau_R * delta_t
                                    + (n[0,:,i] - 3 * n_e[0,:,i]) * f_jk[0,0,:])
        n_e[1,:,i+1] = n_e[1,:,i] + (-n_e[1,:,i] / tau_R * delta_t
                                    + (n[0,:,i] - n_e[1,:,i]) * f_jk[0,1,:]
                                    + (n[1,:,i] - 5 / 3 * n_e[1,:,i])
                                    * f_jk[1,0,:])
        n_e[2,:,i+1] = n_e[2,:,i] + (-n_e[2,:,i] / tau_R * delta_t
                                    + (n[0,:,i] - 3 / 5 * n_e[2,:,i])
                                    * f_jk[0,2,:]
                                    + (n[1,:,i] -  n_e[2,:,i]) * f_jk[1,1,:])
        n_e[3,:,i+1] = n_e[3,:,i] + (-n_e[3,:,i] / tau_R * delta_t
                                    + (n[1,:,i] - 5 / 7 * n_e[3,:,i])
                                    * f_jk[1,2,:])

        P_s += (np.sum((n_e[0,:,i] * q_jk[0,0] 
                       + n_e[1,:,i] * (5 / 6 * q_jk[0,1] + 1 / 6 * q_jk[1,0])
                       + n_e[2,:,i] * (1 / 2 * q_jk[0,2] + 1 / 2 * q_jk[1,1])
                       + n_e[3,:,i] * q_jk[1,2]) * temp_spectrum) / tau_R *
                       delta_t / np.sum(temp_spectrum))
        
        n_e2[0,:,i+1] = n_e2[0,:,i] + (-n_e2[0,:,i] / tau_R * delta_t
                                    + 0.375 * f_jk[0,0,:])
        n_e2[1,:,i+1] = n_e2[1,:,i] + (-n_e2[1,:,i] / tau_R * delta_t
                                    + 0.375 * f_jk[0,1,:]
                                    + 0.625 * f_jk[1,0,:])
        n_e2[2,:,i+1] = n_e2[2,:,i] + (-n_e2[2,:,i] / tau_R * delta_t
                                    + 0.375 * f_jk[0,2,:]
                                    + 0.625 * f_jk[1,1,:])
        n_e2[3,:,i+1] = n_e2[3,:,i] + (-n_e2[3,:,i] / tau_R * delta_t
                                    + 0.625 * f_jk[1,2,:])
        
        P_ns += (np.sum((n_e2[0,:,i] * q_jk[0,0] 
                       + n_e2[1,:,i] * (5 / 6 * q_jk[0,1] + 1 / 6 * q_jk[1,0])
                       + n_e2[2,:,i] * (1 / 2 * q_jk[0,2] + 1 / 2 * q_jk[1,1])
                       + n_e2[3,:,i] * q_jk[1,2]) * temp_spectrum) / tau_R *
                       delta_t / np.sum(temp_spectrum))

    P_s += (np.sum((n_e[0,:,i+1] * q_jk[0,0] 
                       + n_e[1,:,i+1] * (5 / 6 * q_jk[0,1] + 1 / 6 * q_jk[1,0])
                       + n_e[2,:,i+1] * (1 / 2 * q_jk[0,2] + 1 / 2 * q_jk[1,1])
                       + n_e[3,:,i+1] * q_jk[1,2]) * temp_spectrum) / 
                       np.sum(temp_spectrum))
    P_ns += (np.sum((n_e2[0,:,i+1] * q_jk[0,0] 
                       + n_e2[1,:,i+1] * (5 / 6 * q_jk[0,1] + 1 / 6 * q_jk[1,0])
                       + n_e2[2,:,i+1] * (1 / 2 * q_jk[0,2] + 1 / 2 * q_jk[1,1])
                       + n_e2[3,:,i+1] * q_jk[1,2]) * temp_spectrum) / 
                       np.sum(temp_spectrum))
    
    #If ratio=True, the degree of saturation is returned. If ratio != True, the 
    # total number of emitted photons in the case with saturation and without 
    # saturation are returned individually
                       
    if ratio:
        return 1 - P_s / P_ns
    else:
        return P_s, P_ns

def gauss_1D(alpha_L, r):
    #Returns a 1D Gaussian profile
    return np.exp(-4*r**2/alpha_L**2)

def get_saturation_beam(nu_L, Delta_nu_L, N_L, z, T_atm, alpha_L, alpha_T, 
                        t_L=10, nt=50, delta_t=1, delta_r = 5 * 10**-6, 
                        Temp_Na=200, lineshape='gauss', ratio_beam=True):
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
        sats[:,i] = get_saturation(nu_L, Delta_nu_L, beam[i], z, alpha_L, T_atm,
                                   t_L, nt, delta_t, Temp_Na, lineshape, 
                                   ratio=False)
    
    #If ratio_beam == True, the degree of saturation is returned. If
    # ratio_beam != True, the total number of emitted photons in the case with
    # saturation and without saturation are returned individually
                            
    if ratio_beam:
        return 1 - np.sum(sats[0,:] * r) / np.sum(sats[1,:] * r)
    else:
        return np.sum(sats[0,:] * r)/Omega, np.sum(sats[1,:] * r)/Omega
    
def get_wind_and_temp_errors(Temp_Na, nu_Ls, Delta_nu_L, N_L, z, T_atm, alpha_L,
                             alpha_T, t_L=10, nt=50, delta_t=1, delta_r=1e-5, 
                             lineshape='gauss'):
    #Determines the retrieved spectrum, with and without saturation, at the 
    # given measurement frequencies. Fits a spectrum based on the measurements
    # in each case, and returns the fit parameters (amplitude, temperature, and
    # LOS wind speed)
    
    Ps = np.zeros((2,len(nu_Ls)))
    norm = get_total_scattering_cross_section(Temp_Na, 0)
    
    for i in range(len(nu_Ls)):
        nu_L = nu_Ls[i]
        Ps[:,i] = get_saturation_beam(nu_L, Delta_nu_L, N_L, z, T_atm, alpha_L,
                                      alpha_T, t_L, nt, delta_t, delta_r, 
                                      Temp_Na, lineshape, False)
    
    p = np.array([np.max(Ps)/norm, Temp_Na, 1])
    res_sat = opt.leastsq(fit_wind_and_temp, p,
                          args=(nu_Ls, Ps[0,:], Delta_nu_L, lineshape),
                          full_output=1)
    res_no_sat = opt.leastsq(fit_wind_and_temp, p,
                             args=(nu_Ls, Ps[1,:], Delta_nu_L, lineshape),
                             full_output=1)
    
    return res_sat, res_no_sat, Ps
