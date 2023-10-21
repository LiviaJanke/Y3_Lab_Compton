# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:00:37 2023

@author: lme19
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as math
import matplotlib.dates as mdates
from scipy.optimize import curve_fit

from sklearn.linear_model import LinearRegression
from scipy.stats import linregress

#%%
#Defining  functions

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


# Compare measured values with the energies of the scattered photon calculated according to Equation 1

# Equation 1:

def equation_1(theta):
    
    E_gamma_0 = 661.7

    mec2 = 511
    
    E_gamma_theta = E_gamma_0 / (1 + (E_gamma_0 / mec2) * (1 - np.cos(theta)))
    
    return E_gamma_theta

    
def Gauss_res_func(x):
    x0 = 256
    
    a = 1
    
    sigma = 6.370358898274867
    
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def calculate_compton_spectrum(df, background_df):
    
    counts = df['Events_N']
    
    background_counts = background_df['Events_N']
    
    counts_smeared = np.convolve(counts, gauss_res_vals, 'same')
    
    background_counts_smeared = np.convolve(background_counts, gauss_res_vals, 'same')

    compton_smeared = counts_smeared - background_counts_smeared 
    
    return compton_smeared 


#%%


# importing data

deg_45_background_df = pd.read_csv('Angles_200s_meas/45_deg_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_45_v1_df =  pd.read_csv('Angles_200s_meas/45_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_45_v2_df =  pd.read_csv('Angles_200s_meas/45_deg_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_45_v3_df =  pd.read_csv('Angles_200s_meas/45_deg_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

deg_50_background_df = pd.read_csv('Angles_200s_meas/50_deg_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_50_v1_df =  pd.read_csv('Angles_200s_meas/50_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_50_v2_df =  pd.read_csv('Angles_200s_meas/50_deg_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_50_v3_df =  pd.read_csv('Angles_200s_meas/50_deg_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

deg_60_background_df = pd.read_csv('Angles_200s_meas/60_deg_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_60_v1_df =  pd.read_csv('Angles_200s_meas/60_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_60_v2_df =  pd.read_csv('Angles_200s_meas/60_deg_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_60_v3_df =  pd.read_csv('Angles_200s_meas/60_deg_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

deg_70_background_df = pd.read_csv('Angles_200s_meas/70_deg_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_70_v1_df =  pd.read_csv('Angles_200s_meas/70_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_70_v2_df =  pd.read_csv('Angles_200s_meas/70_deg_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_70_v3_df =  pd.read_csv('Angles_200s_meas/70_deg_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

deg_80_background_df = pd.read_csv('Angles_200s_meas/80_deg_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_80_v1_df =  pd.read_csv('Angles_200s_meas/80_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_80_v2_df =  pd.read_csv('Angles_200s_meas/80_deg_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_80_v3_df =  pd.read_csv('Angles_200s_meas/80_deg_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

deg_90_background_300s_df = pd.read_csv('Angles_200s_meas/90_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_90_v1_300s_df =  pd.read_csv('Angles_200s_meas/90_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_90_v2_300s_df =  pd.read_csv('Angles_200s_meas/90_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_90_v3_300s_df =  pd.read_csv('Angles_200s_meas/90_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

deg_90_background_df = pd.read_csv('Angles_200s_meas/90_deg_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_90_v1_df =  pd.read_csv('Angles_200s_meas/90_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_90_v2_df =  pd.read_csv('Angles_200s_meas/90_deg_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_90_v3_df =  pd.read_csv('Angles_200s_meas/90_deg_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

deg_100_background_300s_df = pd.read_csv('Angles_200s_meas/100_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_100_v1_300s_df =  pd.read_csv('Angles_200s_meas/100_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_100_v2_300s_df =  pd.read_csv('Angles_200s_meas/100_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_100_v3_300s_df =  pd.read_csv('Angles_200s_meas/100_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

deg_110_background_300s_df = pd.read_csv('Angles_200s_meas/110_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_110_v1_300s_df =  pd.read_csv('Angles_200s_meas/110_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_110_v2_300s_df =  pd.read_csv('Angles_200s_meas/110_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_110_v3_300s_df =  pd.read_csv('Angles_200s_meas/110_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])


calibrated_energies = np.loadtxt('calibrated_energies_non_and_lin.csv')

#%%

# Smearing input spectra with Gaussian resolution function
# Full example to show that the process works


x = np.linspace(1, 511, 511)


gauss_res_vals = Gauss_res_func(x)


plt.plot(x, gauss_res_vals)
plt.show()

deg_45_background_counts = deg_45_background_df['Events_N']
deg_45_background_channels = deg_45_background_df['channel_n']

deg_45_v1_counts = deg_45_v1_df['Events_N']

deg_45_background_counts_smeared = np.convolve(deg_45_background_counts, gauss_res_vals, 'same')

plt.plot(deg_45_background_channels, deg_45_background_counts)
plt.show()

plt.plot(deg_45_background_channels, deg_45_background_counts_smeared)
plt.show()

deg_45_v1_counts_smeared = np.convolve(deg_45_v1_counts, gauss_res_vals, 'same')

plt.plot(deg_45_background_channels, deg_45_v1_counts)
plt.show()

plt.plot(deg_45_background_channels, deg_45_v1_counts_smeared)
plt.show()

deg_45_compton = deg_45_v1_counts - deg_45_background_counts
deg_45_compton_smeared = deg_45_v1_counts_smeared - deg_45_background_counts_smeared

plt.plot(calibrated_energies, deg_45_compton)
plt.show()

plt.plot(calibrated_energies, deg_45_compton_smeared)
plt.show()

#%%


deg_45_compton_v1 = calculate_compton_spectrum(deg_45_v1_df, deg_45_background_df)
deg_45_compton_v2 = calculate_compton_spectrum(deg_45_v2_df, deg_45_background_df)
deg_45_compton_v3 = calculate_compton_spectrum(deg_45_v3_df, deg_45_background_df)

deg_45_compton = (deg_45_compton_v1 + deg_45_compton_v2 + deg_45_compton_v3) / 3
deg_45_compton_df = pd.DataFrame(deg_45_compton, columns = ['45'])

deg_50_compton_v1 = calculate_compton_spectrum(deg_50_v1_df, deg_50_background_df)
deg_50_compton_v2 = calculate_compton_spectrum(deg_50_v2_df, deg_50_background_df)
deg_50_compton_v3 = calculate_compton_spectrum(deg_50_v3_df, deg_50_background_df)

deg_50_compton = (deg_50_compton_v1 + deg_50_compton_v2 + deg_50_compton_v3) / 3
deg_50_compton_df = pd.DataFrame(deg_50_compton, columns = ['50'])

deg_60_compton_v1 = calculate_compton_spectrum(deg_60_v1_df, deg_60_background_df)
deg_60_compton_v2 = calculate_compton_spectrum(deg_60_v2_df, deg_60_background_df)
deg_60_compton_v3 = calculate_compton_spectrum(deg_60_v3_df, deg_60_background_df)

deg_60_compton = (deg_60_compton_v1 + deg_60_compton_v2 + deg_60_compton_v3) / 3
deg_60_compton_df = pd.DataFrame(deg_60_compton, columns = ['60'])

deg_70_compton_v1 = calculate_compton_spectrum(deg_70_v1_df, deg_70_background_df)
deg_70_compton_v2 = calculate_compton_spectrum(deg_70_v2_df, deg_70_background_df)
deg_70_compton_v3 = calculate_compton_spectrum(deg_70_v3_df, deg_70_background_df)

deg_70_compton = (deg_70_compton_v1 + deg_70_compton_v2 + deg_70_compton_v3) / 3
deg_70_compton_df = pd.DataFrame(deg_70_compton, columns = ['70'])

deg_80_compton_v1 = calculate_compton_spectrum(deg_80_v1_df, deg_80_background_df)
deg_80_compton_v2 = calculate_compton_spectrum(deg_80_v2_df, deg_80_background_df)
deg_80_compton_v3 = calculate_compton_spectrum(deg_80_v3_df, deg_80_background_df)

deg_80_compton = (deg_80_compton_v1 + deg_80_compton_v2 + deg_80_compton_v3) / 3
deg_80_compton_df = pd.DataFrame(deg_80_compton, columns = ['80'])

deg_90_compton_v1 = calculate_compton_spectrum(deg_90_v1_df, deg_90_background_df)
deg_90_compton_v2 = calculate_compton_spectrum(deg_90_v2_df, deg_90_background_df)
deg_90_compton_v3 = calculate_compton_spectrum(deg_90_v3_df, deg_90_background_df)

deg_90_compton = (deg_90_compton_v1 + deg_90_compton_v2 + deg_90_compton_v3) / 3
deg_90_compton_df = pd.DataFrame(deg_90_compton, columns = ['90'])

deg_100_compton_v1 = calculate_compton_spectrum(deg_100_v1_300s_df, deg_100_background_300s_df)
deg_100_compton_v2 = calculate_compton_spectrum(deg_100_v2_300s_df, deg_100_background_300s_df)
deg_100_compton_v3 = calculate_compton_spectrum(deg_100_v3_300s_df, deg_100_background_300s_df)

deg_100_compton = (deg_100_compton_v1 + deg_100_compton_v2 + deg_100_compton_v3) / 3
deg_100_compton_df = pd.DataFrame(deg_100_compton, columns = ['100'])

# calculating mean compton profiles
# without resolution function
#deg_45_compton = ((deg_45_v1_df['Events_N'] - deg_45_background_df['Events_N']) + (deg_45_v2_df['Events_N'] - deg_45_background_df['Events_N']) + (deg_45_v3_df['Events_N'] - deg_45_background_df['Events_N'])) / 3
#plt.plot(calibrated_energies, deg_45_compton)

#deg_50_compton = ((deg_50_v1_df['Events_N'] - deg_50_background_df['Events_N']) + (deg_50_v2_df['Events_N'] - deg_50_background_df['Events_N']) + (deg_50_v3_df['Events_N'] - deg_50_background_df['Events_N'])) / 3
#deg_60_compton = ((deg_60_v1_df['Events_N'] - deg_60_background_df['Events_N']) + (deg_60_v2_df['Events_N'] - deg_60_background_df['Events_N']) + (deg_60_v3_df['Events_N'] - deg_60_background_df['Events_N'])) / 3
#deg_70_compton = ((deg_70_v1_df['Events_N'] - deg_70_background_df['Events_N']) + (deg_70_v2_df['Events_N'] - deg_70_background_df['Events_N']) + (deg_70_v3_df['Events_N'] - deg_70_background_df['Events_N'])) / 3
#deg_80_compton = ((deg_80_v1_df['Events_N'] - deg_80_background_df['Events_N']) + (deg_80_v2_df['Events_N'] - deg_80_background_df['Events_N']) + (deg_80_v3_df['Events_N'] - deg_80_background_df['Events_N'])) / 3
#deg_90_compton = ((deg_90_v1_df['Events_N'] - deg_90_background_df['Events_N']) + (deg_90_v2_df['Events_N'] - deg_90_background_df['Events_N']) + (deg_90_v3_df['Events_N'] - deg_90_background_df['Events_N'])) / 3
#deg_100_compton = ((deg_100_v1_300s_df['Events_N'] - deg_100_background_300s_df['Events_N']) + (deg_100_v2_300s_df['Events_N'] - deg_100_background_300s_df['Events_N']) + (deg_100_v3_300s_df['Events_N'] - deg_100_background_300s_df['Events_N'])) / 3

#%%

compton_arrays_df = pd.concat([deg_45_compton_df, deg_50_compton_df, deg_60_compton_df, deg_70_compton_df, deg_80_compton_df, deg_90_compton_df, deg_100_compton_df], axis = 1,  keys=['45', '50', '60', '70', '80', '90', '100']) 

channels = deg_45_background_df['channel_n']


#%%

calibrated_energies = np.loadtxt('calibrated_energies_non_and_lin.csv')

autocalibrated_energies = deg_45_background_df['Energy_keV']
 

angles_deg = np.array((45, 50, 60, 70, 80, 90, 100))

angles_rad = angles_deg * np.pi / 180

expected_vals = equation_1(angles_rad)


energies_compton = []
uncerts_compton = []

for i in np.arange(0, len(angles_rad)):
    
    angle = angles_deg[i]
    
    print('angle degrees')
    print(angle)
    
    expect  = equation_1(angle * np.pi / 180)
    
    print('expectation value')
    print(expect)
    
    channel_val = (expect - 23.35650629722379) / 1.6445278681687145
    
    
    print('channel value')
    print(channel_val)
    
    energies_fitting_range = calibrated_energies[int(channel_val - 25): int(channel_val + 55)]
    
    energies_fitting_range_df = pd.DataFrame(energies_fitting_range)
    
    print('energies fitting range')
    print(energies_fitting_range)
    
    compton_array = compton_arrays_df[str(angle)] + 7
    
    
    print('comtpon array')
    print(compton_array)
    
    compton_fitting_range_df = compton_array[int(channel_val - 25): int(channel_val + 55)]
    
    compton_fitting_range = compton_fitting_range_df.values.flatten()
    
    print('compton fitting range')
    print(compton_fitting_range)
    
    x = energies_fitting_range

    y = compton_fitting_range
    
    plt.plot(x, y, 'b+:', label='data')
    plt.plot(calibrated_energies, compton_array)
    
    # weighted arithmetic mean 
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
    
    print('mean')
    print(mean)
    print('sigma')
    print(sigma)
    
    popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

    plt.plot(x, y, 'b+:', label='data')
    plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
    plt.legend()
    plt.title(str(angle))
    plt.xlabel('Energy')
    plt.ylabel('Counts')
    plt.show()
    
    uncerts = (np.sqrt(np.diag(pcov)))
    energy_uncert = uncerts[1]
    energy = popt[1]

    energies_compton.append(energy)
    uncerts_compton.append(energy_uncert)
    
#%%

plt.errorbar(angles_deg, energies_compton, yerr = uncerts_compton, label = 'measured')
plt.plot(angles_deg, expected_vals, label = 'expected')
plt.title('Compton Effect Verification')
plt.grid()
plt.legend()
plt.show()

#%%

# 300s measurement series 




















