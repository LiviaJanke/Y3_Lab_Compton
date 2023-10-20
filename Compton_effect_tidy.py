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

#%%


# calculating mean compton profiles

deg_45_compton = ((deg_45_v1_df['Events_N'] - deg_45_background_df['Events_N']) + (deg_45_v2_df['Events_N'] - deg_45_background_df['Events_N']) + (deg_45_v3_df['Events_N'] - deg_45_background_df['Events_N'])) / 3
#plt.plot(calibrated_energies, deg_45_compton)

deg_50_compton = ((deg_50_v1_df['Events_N'] - deg_50_background_df['Events_N']) + (deg_50_v2_df['Events_N'] - deg_50_background_df['Events_N']) + (deg_50_v3_df['Events_N'] - deg_50_background_df['Events_N'])) / 3
deg_60_compton = ((deg_60_v1_df['Events_N'] - deg_60_background_df['Events_N']) + (deg_60_v2_df['Events_N'] - deg_60_background_df['Events_N']) + (deg_60_v3_df['Events_N'] - deg_60_background_df['Events_N'])) / 3
deg_70_compton = ((deg_70_v1_df['Events_N'] - deg_70_background_df['Events_N']) + (deg_70_v2_df['Events_N'] - deg_70_background_df['Events_N']) + (deg_70_v3_df['Events_N'] - deg_70_background_df['Events_N'])) / 3
deg_80_compton = ((deg_80_v1_df['Events_N'] - deg_80_background_df['Events_N']) + (deg_80_v2_df['Events_N'] - deg_80_background_df['Events_N']) + (deg_80_v3_df['Events_N'] - deg_80_background_df['Events_N'])) / 3
deg_90_compton = ((deg_90_v1_df['Events_N'] - deg_90_background_df['Events_N']) + (deg_90_v2_df['Events_N'] - deg_90_background_df['Events_N']) + (deg_90_v3_df['Events_N'] - deg_90_background_df['Events_N'])) / 3
deg_100_compton = ((deg_100_v1_300s_df['Events_N'] - deg_100_background_300s_df['Events_N']) + (deg_100_v2_300s_df['Events_N'] - deg_100_background_300s_df['Events_N']) + (deg_100_v3_300s_df['Events_N'] - deg_100_background_300s_df['Events_N'])) / 3

compton_arrays_df = pd.concat([deg_45_compton, deg_50_compton, deg_60_compton, deg_70_compton, deg_80_compton, deg_90_compton, deg_100_compton], axis = 1,  keys=['45', '50', '60', '70', '80', '90', '100']) 

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
    
    print(angle)
    
    expect  = equation_1(angle * np.pi / 180)
    
    print(expect)
    
    energies_fitting_range = np.linspace(int(expect - 40), int(expect + 40), 80)
    
    print(energies_fitting_range)
    
    compton_array = compton_arrays_df[str(angle)] + 15
    
    print(compton_array)
    
    compton_fitting_range = compton_array[int(expect - 40): int(expect + 40)]
    
    print(compton_fitting_range)
    
    x = energies_fitting_range

    y = compton_fitting_range
    
    plt.plot(x, y, 'b+:', label='data')
    
    # weighted arithmetic mean 
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
    
    print(mean)
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
    energy= popt[1]

    energies_compton.append(energy)
    uncerts_compton.append(energy_uncert)
    




















