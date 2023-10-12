# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:26:20 2023

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

#Defining usaeful functions

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


# Compare measured values with the energies of the scattered photon calculated according to Equation 1

# Equation 1:
    

# what is initial photon energy?

# its the Cs-137 peak energy - 122.0 keV
# mec^2 = 511 keV is  the rest energy of the electron

def equation_1(theta):
    
    E_gamma_0 = 661.7

    mec2 = 511
    
    E_gamma_theta = E_gamma_0 / (1 + (E_gamma_0 / mec2) * (1 - np.cos(theta)))
    
    return E_gamma_theta
    
    


#%%

# importing data

deg_45_background_df = pd.read_csv('Angles_200s_meas/45_deg_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_45_v1_df =  pd.read_csv('Angles_200s_meas/45_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_45_v2_df =  pd.read_csv('Angles_200s_meas/45_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_45_v3_df =  pd.read_csv('Angles_200s_meas/45_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

#deg_50_background_df = pd.read_csv('Angles_200s_meas/50_deg_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

# I didn't save a csv version of the 50 deg backgorund file - need to do this on the lab computer

deg_50_v1_df =  pd.read_csv('Angles_200s_meas/50_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_50_v2_df =  pd.read_csv('Angles_200s_meas/50_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_50_v3_df =  pd.read_csv('Angles_200s_meas/50_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

deg_60_background_df = pd.read_csv('Angles_200s_meas/60_deg_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_60_v1_df =  pd.read_csv('Angles_200s_meas/60_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_60_v2_df =  pd.read_csv('Angles_200s_meas/60_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_60_v3_df =  pd.read_csv('Angles_200s_meas/60_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

deg_70_background_df = pd.read_csv('Angles_200s_meas/70_deg_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_70_v1_df =  pd.read_csv('Angles_200s_meas/70_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_70_v2_df =  pd.read_csv('Angles_200s_meas/70_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_70_v3_df =  pd.read_csv('Angles_200s_meas/70_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

deg_80_background_df = pd.read_csv('Angles_200s_meas/80_deg_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_80_v1_df =  pd.read_csv('Angles_200s_meas/80_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_80_v2_df =  pd.read_csv('Angles_200s_meas/80_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_80_v3_df =  pd.read_csv('Angles_200s_meas/80_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

deg_90_background_df = pd.read_csv('Angles_200s_meas/90_deg_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_90_v1_df =  pd.read_csv('Angles_200s_meas/90_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_90_v2_df =  pd.read_csv('Angles_200s_meas/90_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_90_v3_df =  pd.read_csv('Angles_200s_meas/90_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])



#%%

# Analysing Data

deg_45_background_counts = deg_45_background_df['Events_N']
deg_45_background_channels = deg_45_background_df['channel_n']
deg_45_background_energies = deg_45_background_df['Energy_keV']


deg_45_v1_counts = deg_45_v1_df['Events_N']
deg_45_v1_channels = deg_45_v1_df['channel_n']

deg_45_compton = deg_45_v1_counts - deg_45_background_counts
plt.plot(deg_45_background_energies, deg_45_compton)

#%%

# Skipping 50 for now since no background file

deg_60_background_counts = deg_60_background_df['Events_N']
deg_60_background_channels = deg_60_background_df['channel_n']
deg_60_background_energies = deg_60_background_df['Energy_keV']


deg_60_v1_counts = deg_60_v1_df['Events_N']
deg_60_v1_channels = deg_60_v1_df['channel_n']

deg_60_compton = deg_60_v1_counts - deg_60_background_counts
plt.plot(deg_60_background_energies, deg_60_compton)

# These earlier readings haven't been calibrated to energy, so the energies are actually just channel numbers


#%%

deg_70_background_counts = deg_70_background_df['Events_N']
deg_70_background_channels = deg_70_background_df['channel_n']
deg_70_background_energies = deg_70_background_df['Energy_keV']


deg_70_v1_counts = deg_70_v1_df['Events_N']
deg_70_v1_channels = deg_70_v1_df['channel_n']

deg_70_compton = deg_70_v1_counts - deg_70_background_counts
plt.plot(deg_70_background_energies, deg_70_compton)

#%%

deg_80_background_counts = deg_80_background_df['Events_N']
deg_80_background_channels = deg_80_background_df['channel_n']
deg_80_background_energies = deg_80_background_df['Energy_keV']


deg_80_v1_counts = deg_80_v1_df['Events_N']
deg_60_v1_channels = deg_60_v1_df['channel_n']

deg_80_compton = deg_80_v1_counts - deg_60_background_counts
plt.plot(deg_80_background_energies, deg_80_compton)


#%%

deg_90_background_counts = deg_90_background_df['Events_N']
deg_90_background_channels = deg_90_background_df['channel_n']
deg_90_background_energies = deg_90_background_df['Energy_keV']

deg_90_v1_counts = deg_90_v1_df['Events_N']
deg_90_v1_channels = deg_90_v1_df['channel_n']

deg_90_compton = deg_90_v1_counts - deg_90_background_counts

plt.plot(deg_90_background_energies, deg_90_compton)


#%%

# Expected peak values:
    
expected_45 = equation_1(45)
expected_60 = equation_1(60)
expected_70 = equation_1(70)
expected_80 = equation_1(80)
expected_90 = equation_1(90)

angles = np.array((45, 60, 70, 80, 90))

expected_vals = np.array((expected_45, expected_60, expected_70, expected_80, expected_90))


print(angles)
print(expected_vals)

#%%

# Looking at 45 degrees first, decently strong peak near to expected value

# plt.plot(deg_45_background_energies, deg_45_compton, marker = 'x', linewidth = 0)

# add an offset to make positive

# doesn't really matter as we are only interested in the shape and the x location of the peak


deg_45_compton_offset = deg_45_compton + 10

# want to look at energies from 380 to 520
# corresponding channels: 244, 328


plt.plot(deg_45_background_energies[244:328], deg_45_compton_offset[244:328], marker = 'x', linewidth = 0)

# use these ranges for gaussian fitting

x = deg_45_background_energies[244:328]

y = deg_45_compton_offset[244:328]


# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))



popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Deg 45 v1 gauss fit')
plt.xlabel('Energy')
plt.ylabel('Counts')
plt.show()


uncerts_deg_45 = (np.sqrt(np.diag(pcov)))

energy_deg_45 = popt[1]


energy_deg_45_uncert = uncerts_deg_45[1]


# obtained energy value: 45 kev
# expected energy value: 409.80605808

# hmmmmm
# maybe I'm analysing the wrong peak?
# potentially a gaussian isn't the right form to fit in this case?






















