# -*- coding: utf-8 -*-

import numpy as np

c_light = 2.99792458 * 10**8 #m/s
k_B = 1.38064852 * 10**-23 #J K^-1
h_planck = 6.626*10**-34
N_A = 6.022 * 10**23 #mol^-1
e_elec = 1.60218*10**-19 #C
eps_0 = 8.85419*10**-12 # F/m
m_elec = 9.10938*10**-31 #kg
M_layer = 0.02299 #kg mol^-1
m_layer = M_layer/N_A

lamb0 =  589.158 * 10**-9 #m, in vacuum
nu0 = c_light / lamb0
tau_R = 16.249 #ns
Delta_nu_n = 1 / (2 * np.pi * tau_R * 1e-9)
absorb_coeff = e_elec**2/(4*eps_0 * m_elec * c_light)
sig_D = nu0*np.sqrt(k_B / (m_layer * c_light**2)) 
f_D = 0.6411

delta_nu = 1e6
nu_shifts = np.arange(-3*10**9, 3*10**9, delta_nu)
nus = nu0 + nu_shifts
nv = len(nu_shifts)

def convolve(a,b):
    #Calculates the convolution of two arrays and corrects for the shift in
    # index arising from np.convolve
    conv = np.convolve(a,b,'same')*delta_nu
    conv[:-1] = conv[1:]
    return conv

def get_natural_absorption_line():
    #Returns the scattering cross-section spectrum of the absorption line 
    # (natural linewidth only)
    
    #Need to make sure sufficient resolution is used to resolve absorption line
    if nu_shifts[1]-nu_shifts[0] > Delta_nu_n / 5.:
        print('Error: insufficient spectral resolution to resolve absorption' +
                                                                      'line.')
        return
    
    natural_absorption_line = (absorb_coeff * f_D / np.pi * 
                       Delta_nu_n / 2 / ((nu_shifts)**2 + (Delta_nu_n/2)**2))
    
    return natural_absorption_line

def get_temperature_spectrum(Temp_layer): 
    #Returns the Doppler-broadened distribution for a given temperature, i.e.
    # the relative populations of atoms with a corresponding velocity-induced
    # Doppler shift
    sig_D_temp = sig_D * np.sqrt(Temp_layer)
    return 1 / np.sqrt(2*np.pi) / sig_D_temp * np.exp(-nu_shifts**2 /
                                                            (2*sig_D_temp**2))

def get_doppler_broadened_spectrum(Temp_layer):
    #Returns the Doppler-broadened scattering cross-section spectrum
    spectrum = np.zeros(len(nu_shifts))
    temp_spectrum = get_temperature_spectrum(Temp_layer)
    absorption_spectrum = get_natural_absorption_line()
    spectrum = convolve(absorption_spectrum, temp_spectrum)
    return spectrum

def g_L_gauss(nu_L, Delta_nu_L):
    #Returns a laser profile with a Gaussian profile
    sigma_L = Delta_nu_L / 2.355
    return 1 / sigma_L / np.sqrt(2 * np.pi) * np.exp(-(nu_shifts - nu_L)**2 
                                                       / (2 * sigma_L**2))

def get_effective_absorption_line(nu_L, Delta_nu_L):
    #Returns the effective absorption spectrum, accounting for laser lineshape 
    laser_spectrum = g_L_gauss(nu_L, Delta_nu_L)
    effective_absorption_line = convolve(laser_spectrum,
                                            get_natural_absorption_line())
    return effective_absorption_line

def N_L_from_pulse_energy(E, nu=nu0):
    #Returns the number of photons per pulse for a pulse energy in mJ
    return E/(h_planck*nu) / 1000

def N_t_tophat(nt, delta_t, t_L, N_L):
    #Returns the number of photons emitted per time interval, assuming a tophat
    # laser pulse
    
    t = np.arange(delta_t/2, nt*delta_t + delta_t/2, delta_t)
    n = len(np.where(t < t_L)[0])
    N = np.zeros(len(t))
    N[:n] = N_L / n
    return N
   
def get_saturation_megie(z, alpha_L, t_L, sigma_eff, N_L, T_atm):
    #Returns the expected degree of saturation, according to the Megie approach
    Omega = np.pi / 4 * alpha_L**2    
    t_s = (z**2 * Omega * t_L) / (2 * sigma_eff * N_L * T_atm)
    return 1 - 1 / (1 + tau_R/t_s) * (1 - tau_R/t_L * tau_R / (t_s + tau_R) *
                               (np.exp(-(t_L / tau_R) * (1 + tau_R / t_s))-1)) 

def get_saturation(nu_L, Delta_nu_L, N_L,  z,  alpha_L, T_atm, t_L=10, nt=50,
                   delta_t=1, Temp_layer=200, ratio=False):
    #Returns the expected degree of saturation, according to the VDG approach    
    N = N_t_tophat(nt, delta_t, t_L, N_L) 
    Omega = np.pi / 4 * alpha_L**2    
    temp_spectrum = get_temperature_spectrum(Temp_layer)
    
    try:
        nt = np.where(np.cumsum(N) > 0.99999 * N_L)[0][0] + 2
    except IndexError:
        print('No abridgement possible. Warning: ' + 
              'only {0:.2f} percent of photons accounted for.'.format(
                  np.sum(N)/N_L * 100))    
       
    n = np.zeros((nv, nt))
    n_e = np.zeros((nv, nt))
    n[:,0] = np.ones(nv)

    n_e2 = np.zeros((nv, nt))

    P_s = 0
    P_ns = 0
    
    effective_absorption_line = get_effective_absorption_line(nu_L, Delta_nu_L)
    
    for i in range(nt - 1):
        number_of_photons = T_atm * N[i] / z**2 / Omega
        f_e = effective_absorption_line * number_of_photons 
        
        if np.max(f_e) > 0.1:
            print('Error: delta_t must be reduced. max_fj_k = {0:.3f}'.format(
                np.max(f_e)))
            return
           
        n[:,i+1] = n[:,i] + (n_e[:,i] / tau_R * delta_t -
                                          (n[:,i] - n_e[:,i]) * f_e)
        n_e[:,i+1] = n_e[:,i] + (-n_e[:,i] / tau_R * delta_t +
                                          (n[:,i] - n_e[:,i]) * f_e)
        
        P_s += (np.sum(n_e[:,i] * temp_spectrum) / tau_R * delta_t /
                                          np.sum(temp_spectrum))

        n_e2[:,i+1] = n_e2[:,i] + (-n_e2[:,i] / tau_R * delta_t + f_e) 
        P_ns += (np.sum(n_e2[:,i] * temp_spectrum) / tau_R * delta_t / 
                                          np.sum(temp_spectrum))
    
    P_s += np.sum(n_e[:,i+1] * temp_spectrum) / np.sum(temp_spectrum)
    P_ns += np.sum(n_e2[:,i+1] * temp_spectrum) / np.sum(temp_spectrum)

    
    return 1 - P_s / P_ns
