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
deg_45_v2_df =  pd.read_csv('Angles_200s_meas/45_deg_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_45_v3_df =  pd.read_csv('Angles_200s_meas/45_deg_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

#deg_50_background_df = pd.read_csv('Angles_200s_meas/50_deg_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

# I didn't save a csv version of the 50 deg backgorund file - need to do this on the lab computer

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

deg_90_background_df = pd.read_csv('Angles_200s_meas/90_deg_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_90_v1_df =  pd.read_csv('Angles_200s_meas/90_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_90_v2_df =  pd.read_csv('Angles_200s_meas/90_deg_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_90_v3_df =  pd.read_csv('Angles_200s_meas/90_deg_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])


deg_90_background_300s_df = pd.read_csv('Angles_200s_meas/90_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_90_v1_300s_df =  pd.read_csv('Angles_200s_meas/90_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_90_v2_300s_df =  pd.read_csv('Angles_200s_meas/90_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_90_v3_300s_df =  pd.read_csv('Angles_200s_meas/90_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

deg_100_background_300s_df = pd.read_csv('Angles_200s_meas/100_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_100_v1_300s_df =  pd.read_csv('Angles_200s_meas/100_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_100_v2_300s_df =  pd.read_csv('Angles_200s_meas/100_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_100_v3_300s_df =  pd.read_csv('Angles_200s_meas/100_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

#deg_110_background_300s_df = pd.read_csv('Angles_200s_meas/110_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_110_v1_300s_df =  pd.read_csv('Angles_200s_meas/110_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_110_v2_300s_df =  pd.read_csv('Angles_200s_meas/110_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
#deg_110_v3_300s_df =  pd.read_csv('Angles_200s_meas/110_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

# No background file measured for 110 degrees - need to go back and record this



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

deg_90_background_300s_counts = deg_90_background_300s_df['Events_N']
deg_90_background_300s_channels = deg_90_background_300s_df['channel_n']
deg_90_background_300s_energies = deg_90_background_300s_df['Energy_keV']

deg_90_v1_300s_counts = deg_90_v1_300s_df['Events_N']
deg_90_v1_300s_channels = deg_90_v1_300s_df['channel_n']

deg_90_300s_compton = deg_90_v1_300s_counts - deg_90_background_300s_counts

plt.plot(deg_90_background_300s_energies, deg_90_300s_compton)

# seems uncalibrated - apply a manual calibration to this dataset

#%%

deg_100_background_300s_counts = deg_100_background_300s_df['Events_N']
deg_100_background_300s_channels = deg_100_background_300s_df['channel_n']
deg_100_background_300s_energies = deg_100_background_300s_df['Energy_keV']

deg_100_v1_300s_counts = deg_100_v1_300s_df['Events_N']
deg_100_v1_300s_channels = deg_100_v1_300s_df['channel_n']

deg_100_300s_compton = deg_100_v1_300s_counts - deg_100_background_300s_counts

plt.plot(deg_90_background_energies, deg_100_300s_compton)

#%%

# No background file for deg 110


#deg_110_background_300s_counts = deg_110_background_300s_df['Events_N']
#deg_110_background_300s_channels = deg_110_background_300s_df['channel_n']
#deg_110_background_300s_energies = deg_110_background_300s_df['Energy_keV']

#deg_110_v1_300s_counts = deg_110_v1_300s_df['Events_N']
#deg_110_v1_300s_channels = deg_110_v1_300s_df['channel_n']

#deg_90_300s_compton = deg_90_v1_300s_counts - deg_90_background_300s_counts

#plt.plot(deg_90_background_300s_energies, deg_90_300s_compton)

#%%

# Expected peak values:
    
expected_45 = equation_1(45 * np.pi / 180)
expected_50 = equation_1(50 * np.pi / 180)
expected_60 = equation_1(60 * np.pi / 180)
expected_70 = equation_1(70 * np.pi / 180)
expected_80 = equation_1(80 * np.pi / 180)
expected_90 = equation_1(90 * np.pi / 180)
expected_100 = equation_1(100 * np.pi / 180)
expected_110 = equation_1(110 * np.pi / 180)

angles = np.array((45, 60, 70, 80, 90, 100))

expected_vals = np.array((expected_45, expected_60, expected_70, expected_80, expected_90, expected_100))


print(angles)
print(expected_vals)

#%%

# Looking at 45 degrees first, decently strong peak near to expected value

plt.plot(deg_45_background_energies, deg_45_compton, marker = 'x', linewidth = 0)

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

# 45 isn't actually auto - calibrated ; the background energies file is calibrated
# but the v1 file isn@t 
# shouldn't really makea difference since channel numbers aren't part of the fit?

#%%

# looking at deg 60 data


deg_60_compton_offset = deg_60_compton + 10

plt.plot(deg_45_background_energies, deg_60_compton_offset, marker = 'x', linewidth = 0)



# want to look at energies from 300 to 420
# corresponding channels: 195, 260

plt.plot(deg_45_background_energies[200:280], deg_60_compton_offset[200:280], marker = 'x', linewidth = 0)

# No particularly recognisable Gaussian in this dataset
# especially not around a value that makes sense
# expected is 180, actual peak is close to 40
# skipping for now 

# use these ranges for gaussian fitting

x = deg_45_background_energies[200:280]
y = deg_60_compton_offset[200:280]


# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))



popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Deg 60 v1 gauss fit')
plt.xlabel('Energy')
plt.ylabel('Counts')
plt.show()


uncerts_deg_60 = (np.sqrt(np.diag(pcov)))

energy_deg_60 = popt[1]


energy_deg_60_uncert = uncerts_deg_60[1]


#%%

deg_70_compton_offset = deg_70_compton + 15

# Looking at 45 degrees first, decently strong peak near to expected value

plt.plot(deg_70_background_energies, deg_70_compton_offset, marker = 'x', linewidth = 0)




# want to look at energies from 250 to 400
# corresponding channels: 165, 256

plt.plot(deg_70_background_energies[165:280], deg_70_compton_offset[165:280], marker = 'x', linewidth = 0)

# use these ranges for gaussian fitting

x = deg_70_background_energies[165:280]

y = deg_70_compton_offset[165:280]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))



popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Deg 70 v1 gauss fit')
plt.xlabel('Energy')
plt.ylabel('Counts')
plt.show()


uncerts_deg_70 = (np.sqrt(np.diag(pcov)))

energy_deg_70 = popt[1]


energy_deg_70_uncert = uncerts_deg_70[1]


#%%


deg_80_compton_offset = deg_80_compton + 15

# Looking at 45 degrees first, decently strong peak near to expected value

plt.plot(deg_80_background_energies, deg_80_compton_offset, marker = 'x', linewidth = 0)


# want to look at energies from 220 to 400
# corresponding channels: 147 , 256

plt.plot(deg_80_background_energies[140:280], deg_80_compton_offset[140:280], marker = 'x', linewidth = 0)

# use these ranges for gaussian fitting

x = deg_80_background_energies[140:280]

y = deg_80_compton_offset[140:280]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))



popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Deg 80 v1 gauss fit')
plt.xlabel('Energy')
plt.ylabel('Counts')
plt.show()


uncerts_deg_80 = (np.sqrt(np.diag(pcov)))

energy_deg_80 = popt[1]


energy_deg_80_uncert = uncerts_deg_80[1]

#%%


deg_90_compton_offset = deg_90_compton + 15

# Looking at 45 degrees first, decently strong peak near to expected value

plt.plot(deg_90_background_energies, deg_90_compton_offset, marker = 'x', linewidth = 0)

# want to look at energies from 200 to 400
# corresponding channels: 135, 256

plt.plot(deg_90_background_energies[130:250], deg_90_compton_offset[130:250], marker = 'x', linewidth = 0)

# use these ranges for gaussian fitting

x = deg_90_background_energies[130:250]

y = deg_90_compton_offset[130:250]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Deg 70 v1 gauss fit')
plt.xlabel('Energy')
plt.ylabel('Counts')
plt.show()


uncerts_deg_90 = (np.sqrt(np.diag(pcov)))

energy_deg_90 = popt[1]

energy_deg_90_uncert = uncerts_deg_90[1]

#%%

deg_100_300s_compton_offset = deg_100_300s_compton + 15

# Looking at 45 degrees first, decently strong peak near to expected value

plt.plot(deg_45_background_energies, deg_100_300s_compton_offset, marker = 'x', linewidth = 0)

# want to look at energies from 200 to 300
# corresponding channels: 135, 195

plt.plot(deg_45_background_energies[140:210], deg_100_300s_compton_offset[140:210], marker = 'x', linewidth = 0)

# use these ranges for gaussian fitting

x = deg_45_background_energies[140:210]

y = deg_100_300s_compton_offset[140:210]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))



popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Deg 100 v1 gauss fit')
plt.xlabel('Energy')
plt.ylabel('Counts')
plt.show()


uncerts_deg_100 = (np.sqrt(np.diag(pcov)))

energy_deg_100 = popt[1]


energy_deg_100_uncert = uncerts_deg_100[1]


#%%

# plotting measured compton vals against angles

measured_energies = np.array((energy_deg_45, energy_deg_60, energy_deg_70, energy_deg_80, energy_deg_90, energy_deg_100))

measured_energies_uncerts = np.array((energy_deg_45_uncert, energy_deg_60_uncert, energy_deg_70_uncert, energy_deg_80_uncert, energy_deg_90_uncert, energy_deg_100_uncert))

plt.errorbar(angles, measured_energies, yerr = measured_energies_uncerts, label = 'measured')

plt.plot(angles, expected_vals, label = 'expected')
plt.title('Compton Effect Verification')
plt.grid()
plt.legend()
plt.show()

#%%
