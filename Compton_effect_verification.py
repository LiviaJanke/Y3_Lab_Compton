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

deg_45_background_counts = deg_45_background_df['Events_N']
deg_45_background_channels = deg_45_background_df['channel_n']
deg_45_background_energies = deg_45_background_df['Energy_keV']

deg_45_v1_df =  pd.read_csv('Angles_200s_meas/45_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

deg_45_v1_counts = deg_45_v1_df['Events_N']
deg_45_v1_channels = deg_45_v1_df['channel_n']

deg_45_compton = deg_45_v1_counts - deg_45_background_counts


plt.plot(deg_45_background_channels, deg_45_compton)

#%%

deg_90_background_df = pd.read_csv('Angles_200s_meas/90_deg_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

deg_90_background_counts = deg_90_background_df['Events_N']
deg_90_background_channels = deg_90_background_df['channel_n']
deg_90_background_energies = deg_90_background_df['Energy_keV']


deg_90_v1_df =  pd.read_csv('Angles_200s_meas/90_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

deg_90_v1_counts = deg_90_v1_df['Events_N']
deg_90_v1_channels = deg_90_v1_df['channel_n']

deg_90_compton = deg_90_v1_counts - deg_90_background_counts


plt.plot(deg_90_background_energies, deg_90_compton)






