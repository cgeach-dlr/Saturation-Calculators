# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as opt
import scipy.interpolate as si
    
c_light = 2.99792458 * 10**8 #m/s
k_B = 1.38064852 * 10**-23 #J K^-1
h_planck = 6.626*10**-34
N_A = 6.022 * 10**23 #mol^-1
e_elec = 1.60218*10**-19 #C
eps_0 = 8.85419*10**-12 # F/m
m_elec = 9.10938*10**-31 #kg
M_Fe = 0.055845 #kg mol^-1
m_Fe = M_Fe/N_A

lamb0_array =  np.array([372.0993e-9, 373.8194e-9, 386.10055e-9]) #m, in vac.
nu0_array = c_light / lamb0_array
A_i_array = np.array([16.2e6, 14.1e6, 9.69e6]) #Hz
tau_R_array = 1 / A_i_array * 1e9 #ns
Delta_nu_n_array = A_i_array / (2 * np.pi) #Hz
absorb_coeff = e_elec**2/(4*eps_0 * m_elec * c_light)
sig_D_array = nu0_array*np.sqrt(k_B / (m_Fe * c_light**2))
f_D_array = np.array([0.0411, 0.0381, 0.0217])

nu_54_372 = -726.5e6 #Hz - relative frequency shift
nu_56_372 = 0 #Hz - relative frequency shift
nu_58_372 = 365.1e6 #Hz - relative frequency shift
nu_59_372 = 689.9e6 #Hz - relative frequency shift 
nu_54_374 = -728.4e6 #Hz - relative frequency shift
nu_56_374 = 0 #Hz - relative frequency shift
nu_58_374 = 365.5e6 #Hz - relative frequency shift
nu_59_374 = 693.6e6 #Hz - relative frequency shift 
nu_54_386 = -743e6 #Hz - relative frequency shift
nu_56_386 = 0 #Hz - relative frequency shift
nu_58_386 = 375e6 #Hz - relative frequency shift
nu_59_386 = 714e6 #Hz - relative frequency shift 

nu_array = np.array([[nu_54_372, nu_56_372, nu_58_372, nu_59_372],
                      [nu_54_374, nu_56_374, nu_58_374, nu_59_374],
                      [nu_54_386, nu_56_386, nu_58_386, nu_59_386]])

p_54 = 0.05845
p_56 = 0.91754
p_58 = 0.02119
p_59 = 0.00282

p_isos_array = np.array([p_54, p_56, p_58, p_59])

R_B = np.array([1,0.9114,1])

#Hanle factors - assumed to be 1 here.
q_jk = np.array([1, 1, 1]) 

delta_nu = 2e5
nu_shifts = np.arange(-4*10**9, 4*10**9, delta_nu)
nv = len(nu_shifts)

def convolve(a,b):
    #Calculates the convolution of two arrays and corrects for the shift in
    # index arising from np.convolve
    
    conv = np.convolve(a,b,'same')*delta_nu
    conv[:-1] = conv[1:]
    return conv

def get_natural_absorption_line(line, iso):
    #Returns the scattering cross-section spectrum of the given absorption
    # line for the given isotope (natural linewidth only)
    
    #Check that sufficient spectral resolution is used to resolve the
    # absorption line
    
    if delta_nu > Delta_nu_n_array[line] / 5.:
        print('Error: insufficient spectral resolution to resolve' + 
              ' absorption line.')
        return
        
    nu_offset = nu_array[line, iso]
    
    natural_absorption_line = (absorb_coeff * f_D_array[line] / np.pi
                               * Delta_nu_n_array[line] / 2 
                               / ((nu_shifts - nu_offset)**2
                                   + (Delta_nu_n_array[line]/2)**2))
    
    return natural_absorption_line

def get_combined_absorption_line(line):
    #Returns the combined scattering cross-section spectrum of the given
    # absorption line for all isotpes, accounting for relative abundances
    
    combined_line = np.zeros(nv)
    for i in range(4):
        combined_line += (p_isos_array[i] 
                          * get_natural_absorption_line(line, i))
    return combined_line

def get_temperature_spectrum(Temp_Fe, line): 
    #Returns the Doppler-broadened distribution for a given temperature, i.e.
    # the relative populations of atoms with a corresponding velocity-induced
    # Doppler shift
    
    sig_D_temp = sig_D_array[line] * np.sqrt(Temp_Fe)
    return 1 / np.sqrt(2*np.pi) / sig_D_temp * np.exp(-nu_shifts**2
                                                      / (2*sig_D_temp**2))

def get_doppler_broadened_spectrum_complete(Temp_Fe, line):
    #Returns the combined Doppler-broadened scattering cross-section spectrum 
    # for a given absorption line, accounting for all isotopes
    
    temp_spectrum = get_temperature_spectrum(Temp_Fe, line)
    combined_spectrum = get_combined_absorption_line(line)
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

def get_effective_absorption_lines(line, nu_L=0, Delta_nu_L = 100*10**6,
                                   lineshape='gauss'):
    #Returns the effective absorption spectra, accounting for laser lineshape

    #Check that sufficient spectral resolution is used to resolve the
    # laser line
                                       
    if delta_nu > Delta_nu_L / 5.:
        print('Error: insufficient spectral resolution to resolve the' + 
              ' laser line.')
        return
        
    L_jk = np.zeros((4, len(nu_shifts)))
    laser_spectrum = get_laser_pulseshape(nu_L, Delta_nu_L, lineshape)
    
    for i in range(4):
        alpha_jk = absorb_coeff * (f_D_array[line] / np.pi 
                                   * Delta_nu_n_array[line] / 2
                                   / ((nu_shifts + nu_array[line, i])**2
                                      + (Delta_nu_n_array[line]/2)**2))
        L_jk[i] = convolve(laser_spectrum, alpha_jk)
    return L_jk

def get_total_scattering_cross_section_spectrum(Temp_Fe, line,
                                                Delta_nu_L=100e6,
                                                lineshape='gauss'):
    #Returns the total effective Doppler-broadened scattering cross-section
    # spectrum, accounting for the laser lineshape
                                                    
    temp_spectrum = get_doppler_broadened_spectrum_complete(Temp_Fe, line)
    laser_spectrum = get_laser_pulseshape(0, Delta_nu_L, lineshape)
    sigma_tot = convolve(laser_spectrum, temp_spectrum)
    return sigma_tot

def get_total_scattering_cross_section(Temp_Fe, line, nu_L=0,
                                       Delta_nu_L=100e6, lineshape='gauss'):
    #Returns the total effective Doppler-broadened scattering cross-section
    # for a given laser frequency
    
    sigma_tot = get_total_scattering_cross_section_spectrum(Temp_Fe, line,
                                                            Delta_nu_L,
                                                            lineshape)
    f_sigma = si.interp1d(nu_shifts, sigma_tot)
    return f_sigma(nu_L)

def fit_wind_and_temp(params, nu_Ls, ydata, line, Delta_nu_L=100e6,
                      lineshape='gauss'):
    #Fit function for a multi-frequency measurement of the scattering 
    # cross-section spectrum, in terms of an amplitude (params[0]), 
    # temperature (params[1]) and line-of-sight wind velocity (params[2])
    
    fit = params[0]*get_total_scattering_cross_section(params[1], line,
                nu_Ls - params[2]/lamb0_array[line], Delta_nu_L, lineshape)
    return fit - ydata

def N_L_from_pulse_energy(E, line): 
    #Returns the number of photons per pulse for a pulse energy in mJ
    
    return E/(h_planck*nu0_array[line]) / 1000

def N_t_laser(nt, delta_t, t_L, N_L):
    #Returns the number of photons emitted per time interval, assuming the
    # laser pulse shape approximation given by VDG
    
    t = np.arange(delta_t/2, nt*delta_t + delta_t/2, delta_t)
    a = 3.4 / t_L 
    return N_L * t**2 * np.exp(-a*t) * a**3 / 2 * delta_t

def get_saturation_megie(z, alpha_L, t_L, sigma_eff, N_L, T_atm, line):
    #Returns the expected degree of saturation, according to the Megie
    # approach
    
    Omega = np.pi / 4 * alpha_L**2  
    tau_R = tau_R_array[line]
    t_s = (z**2 * Omega * t_L) / (2 * sigma_eff * N_L * T_atm)
    return 1 - 1 / (1 + tau_R/t_s) * (1 - tau_R/t_L * tau_R / (t_s + tau_R)
                    * (np.exp(-(t_L / tau_R) * (1 + tau_R / t_s))-1))

def get_saturation(line, nu_L, Delta_nu_L, N_L, z, alpha_L, T_atm, t_L=10,
                   nt=50, delta_t=1, Temp_Fe=1000, lineshape='gauss',
                   ratio=False):
    #Returns the expected degree of saturation, according to the VDG approach    
    
    N = N_t_laser(nt, delta_t, t_L, N_L) 
    Omega = np.pi / 4 * alpha_L**2    
    temp_spectrum = get_temperature_spectrum(Temp_Fe, line)
    
    #Find the index for which 99.999% of photons have been accounted for, in
    # order to abridge the calculation (the DES can be solved analytically for
    # time steps where the number of photons is approximately zero)
    try:
        nt = np.where(np.cumsum(N) > 0.99999 * N_L)[0][0] + 2
    except IndexError:
        print('No abridgement possible. Warning:' +  
              'only {0:.2f} percent of photons accounted for.'.format(
              np.sum(N)/N_L * 100))    
    
    n = np.zeros((4, nv, nt))
    n_e = np.zeros((4, nv, nt))
    
    n[:,:,0] = np.ones(nv) 

    n_e2 = np.zeros((4, nv, nt))

    P_s = 0
    P_ns = 0
    
    L_jk = get_effective_absorption_lines(line, nu_L, Delta_nu_L, lineshape)
    tau_R = tau_R_array[line]
    
    for i in range(nt - 1):
        number_of_photons = T_atm * N[i] / z**2 / Omega
        f_jk = L_jk * number_of_photons
        
        if np.max(f_jk) > 0.1:
            print('Error: delta_t must be reduced. max_f_jk = {0:.3f}'.format(
                  np.max(f_jk)))
            return
        
        n[:,:,i+1] = n[:,:,i] + (R_B[line] * n_e[:,:,i] / tau_R  * delta_t
                                - (n[:,:,i] - n_e[:,:,i]) * f_jk)
        n_e[:,:,i+1] = n_e[:,:,i] + (-n_e[:,:,i] / tau_R * delta_t
                                    + (n[:,:,i] - n_e[:,:,i]) * f_jk)
        
        n_e2[:,:,i+1] = n_e2[:,:,i] - n_e2[:,:,i] / tau_R * delta_t + f_jk
        
        for j in range(4):
            P_s += (R_B[line] * p_isos_array[j] * (np.sum(q_jk[line] * 
                    n_e[j,:,i] * temp_spectrum) / tau_R * delta_t / 
                    np.sum(temp_spectrum)))
            P_ns += (R_B[line] * p_isos_array[j] * (np.sum(q_jk[line] * 
                     n_e2[j,:,i] * temp_spectrum) / tau_R * delta_t /
                     np.sum(temp_spectrum)))
        
    for j in range(4):
        P_s += (R_B[line] * p_isos_array[j] * np.sum(q_jk[line] * 
                n_e[j,:,i+1] * temp_spectrum / np.sum(temp_spectrum)))
        P_ns += (R_B[line] * p_isos_array[j] * np.sum(q_jk[line] *
                 n_e2[j,:,i+1] * temp_spectrum) / np.sum(temp_spectrum))

    #If ratio == True, the degree of saturation is returned. If ratio != True,
    # the total number of emitted photons in the case with saturation and
    # without saturation are returned individually
    if ratio:
        return 1 - P_s / P_ns
    else:
        return P_s, P_ns

def gauss_1D(alpha_L, r):
    #Returns a 1D Gaussian profile
    return np.exp(-4*r**2/alpha_L**2)

def get_saturation_beam(line, nu_L, Delta_nu_L, N_L, z, T_atm, alpha_L,
                        alpha_T, t_L=10, nt=50, delta_t=1, delta_r = 5e-6,
                        Temp_Fe=200, lineshape='gauss', ratio_beam=True):
    #Returns the expected degree of saturation, according to the VDG approach,
    # averaged over a Gaussian beam profile over the field-of-view of the
    # telescope.
    
    #In order to avoid coverage issues, the parameter delta_r is adjusted so
    # that the field-of-view of the instrument is covered by an integer
    # number of equal-width bins.
    
    r_max = alpha_T/2.
    n_r = int(r_max / delta_r)+1
    delta_r_adjusted = r_max / n_r
    
    Omega = np.pi / 4 * alpha_L**2
    
    r = np.arange(delta_r_adjusted/2, r_max, delta_r_adjusted)

    sats = np.zeros((2,len(r)))
    beam = N_L * gauss_1D(alpha_L, r)
    
    for i in range(len(r)):
        sats[:,i] = get_saturation(line, nu_L, Delta_nu_L, beam[i], z,
                                   alpha_L, T_atm, t_L, nt, delta_t, Temp_Fe,
                                   lineshape, ratio=False)
    
    #If ratio_beam == True, the degree of saturation is returned. If 
    # ratio_beam != True, the total number of emitted photons in the case with
    # saturation and without saturation are returned individually
    
    if ratio_beam:
        return 1 - np.sum(sats[0,:] * r) / np.sum(sats[1,:] * r)
    else:
        return np.sum(sats[0,:] * r)/Omega, np.sum(sats[1,:] * r)/Omega
    
def get_wind_and_temp_errors(line, Temp_Fe, nu_Ls, Delta_nu_L, N_L, z, T_atm,
                             alpha_L, alpha_T, t_L=10, nt=50, delta_t=1,
                             delta_r=1e-5, lineshape='gauss'):
    #Determines the retrieved spectrum, with and without saturation, at the
    # given measurement frequencies. Fits a spectrum based on the measurements
    # in each case, and returns the fit parameters (amplitude, temperature, and
    # LOS wind speed)
    
    Ps = np.zeros((2,len(nu_Ls)))
    norm = get_total_scattering_cross_section(Temp_Fe, line)
    
    for i in range(len(nu_Ls)):
        nu_L = nu_Ls[i]
        Ps[:,i] = get_saturation_beam(line, nu_L, Delta_nu_L, N_L, z, T_atm,
                                      alpha_L, alpha_T, t_L, nt, delta_t, 
                                      delta_r, Temp_Fe, lineshape, False)
    
    p = np.array([np.max(Ps)/norm, Temp_Fe, 1])
    res_sat = opt.leastsq(fit_wind_and_temp, p,
                          args=(nu_Ls, Ps[0,:], line, Delta_nu_L, lineshape),
                          full_output=1)
    res_no_sat = opt.leastsq(fit_wind_and_temp, p,
                             args=(nu_Ls, Ps[1,:], line, Delta_nu_L, lineshape),
                             full_output=1)
    
    return res_sat, res_no_sat, Ps
