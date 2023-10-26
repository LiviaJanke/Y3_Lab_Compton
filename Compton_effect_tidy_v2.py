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

    

def calculate_compton_spectrum(df, background_df):
    
    counts = df['Events_N']
    
    background_counts = background_df['Events_N']
    
    compton = counts - background_counts
    
    return compton


#%%


# importing data

deg_10_background_300s_df = pd.read_csv('Angles_200s_meas/10_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_10_v1_300s_df =  pd.read_csv('Angles_200s_meas/10_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_10_v2_300s_df =  pd.read_csv('Angles_200s_meas/10_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_10_v3_300s_df =  pd.read_csv('Angles_200s_meas/10_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

deg_20_background_300s_df = pd.read_csv('Angles_200s_meas/20_deg_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_20_v1_300s_df =  pd.read_csv('Angles_200s_meas/20_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_20_v2_300s_df =  pd.read_csv('Angles_200s_meas/20_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_20_v3_300s_df =  pd.read_csv('Angles_200s_meas/20_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

deg_30_background_300s_df = pd.read_csv('Angles_200s_meas/30_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_30_v1_300s_df =  pd.read_csv('Angles_200s_meas/30_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_30_v2_300s_df =  pd.read_csv('Angles_200s_meas/30_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_30_v3_300s_df =  pd.read_csv('Angles_200s_meas/30_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

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

deg_120_background_300s_df = pd.read_csv('Angles_200s_meas/120_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_120_v1_300s_df =  pd.read_csv('Angles_200s_meas/120_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_120_v2_300s_df =  pd.read_csv('Angles_200s_meas/120_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_120_v3_300s_df =  pd.read_csv('Angles_200s_meas/120_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

deg_130_background_300s_df = pd.read_csv('Angles_200s_meas/130_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_130_v1_300s_df =  pd.read_csv('Angles_200s_meas/130_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_130_v2_300s_df =  pd.read_csv('Angles_200s_meas/130_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_130_v3_300s_df =  pd.read_csv('Angles_200s_meas/130_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

deg_140_background_300s_df = pd.read_csv('Angles_200s_meas/110_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_140_v1_300s_df =  pd.read_csv('Angles_200s_meas/110_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_140_v2_300s_df =  pd.read_csv('Angles_200s_meas/110_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg_140_v3_300s_df =  pd.read_csv('Angles_200s_meas/110_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])



calibrated_energies = np.loadtxt('calibrated_energies_non_and_lin.csv')

#%%

# Mean counts for each degree

deg_10_counts = (deg_10_v1_300s_df['Events_N'] + deg_10_v2_300s_df['Events_N'] + deg_10_v3_300s_df['Events_N']) / 3
deg_10_compton = deg_10_counts - deg_10_background_300s_df['Events_N']
deg_10_compton_df = pd.DataFrame(data = [deg_10_compton], columns = ['10'])

deg_20_counts = (deg_20_v1_300s_df['Events_N'] + deg_20_v2_300s_df['Events_N'] + deg_20_v3_300s_df['Events_N']) / 3
deg_20_compton = deg_20_counts - deg_20_background_300s_df['Events_N']
deg_20_compton_df = pd.DataFrame(data = [deg_20_compton], columns = ['20'])

deg_30_counts = (deg_30_v1_300s_df['Events_N'] + deg_30_v2_300s_df['Events_N'] + deg_30_v3_300s_df['Events_N']) / 3
deg_30_compton = deg_30_counts - deg_30_background_300s_df['Events_N']
deg_30_compton_df = pd.DataFrame(data = [deg_30_compton], columns = ['30'])

deg_45_counts = (deg_45_v1_df['Events_N'] + deg_45_v2_df['Events_N'] + deg_45_v3_df['Events_N']) / 3
deg_45_compton = deg_45_counts - deg_45_background_df['Events_N']
deg_45_compton_df = pd.DataFrame(data = [deg_45_compton], columns = ['45'])

deg_50_counts = (deg_50_v1_df['Events_N'] + deg_50_v2_df['Events_N'] + deg_50_v3_df['Events_N']) / 3
deg_50_compton = deg_50_counts - deg_50_background_df['Events_N']
deg_50_compton_df = pd.DataFrame(data = [deg_50_compton], columns = ['50'])

deg_60_counts = (deg_60_v1_df['Events_N'] + deg_60_v2_df['Events_N'] + deg_60_v3_df['Events_N']) / 3
deg_60_compton = deg_60_counts - deg_60_background_df['Events_N']
deg_60_compton_df = pd.DataFrame(data = [deg_60_compton], columns = ['60'])

deg_60_counts = (deg_60_v1_df['Events_N'] + deg_60_v2_df['Events_N'] + deg_60_v3_df['Events_N']) / 3
deg_60_compton = deg_60_counts - deg_60_background_df['Events_N']
deg_60_compton_df = pd.DataFrame(data = [deg_60_compton], columns = ['60'])

deg_70_counts = (deg_70_v1_df['Events_N'] + deg_70_v2_df['Events_N'] + deg_70_v3_df['Events_N']) / 3
deg_70_compton = deg_70_counts - deg_70_background_df['Events_N']
deg_70_compton_df = pd.DataFrame(data = [deg_70_compton], columns = ['70'])

deg_80_counts = (deg_80_v1_df['Events_N'] + deg_80_v2_df['Events_N'] + deg_80_v3_df['Events_N']) / 3
deg_80_compton = deg_80_counts - deg_80_background_df['Events_N']
deg_80_compton_df = pd.DataFrame(data = [deg_80_compton], columns = ['80'])

deg_90_counts = (deg_90_v1_df['Events_N'] + deg_90_v2_df['Events_N'] + deg_90_v3_df['Events_N']) / 3
deg_90_compton = deg_90_counts - deg_90_background_df['Events_N']
deg_90_compton_df = pd.DataFrame(data = [deg_90_compton], columns = ['90'])

deg_100_counts = (deg_100_v1_300s_df['Events_N'] + deg_100_v2_300s_df['Events_N'] + deg_100_v3_300s_df['Events_N']) / 3
deg_100_compton = deg_100_counts - deg_100_background_300s_df['Events_N']
deg_100_compton_df = pd.DataFrame(data = [deg_100_compton], columns = ['100'])

deg_110_counts = (deg_110_v1_300s_df['Events_N'] + deg_110_v2_300s_df['Events_N'] + deg_110_v3_300s_df['Events_N']) / 3
deg_110_compton = deg_110_counts - deg_110_background_300s_df['Events_N']
deg_110_compton_df = pd.DataFrame(data = [deg_110_compton], columns = ['110'])

deg_120_counts = (deg_120_v1_300s_df['Events_N'] + deg_120_v2_300s_df['Events_N'] + deg_120_v3_300s_df['Events_N']) / 3
deg_120_compton = deg_120_counts - deg_120_background_300s_df['Events_N']
deg_120_compton_df = pd.DataFrame(data = [deg_120_compton], columns = ['120'])

deg_130_counts = (deg_130_v1_300s_df['Events_N'] + deg_130_v2_300s_df['Events_N'] + deg_130_v3_300s_df['Events_N']) / 3
deg_130_compton = deg_130_counts - deg_130_background_300s_df['Events_N']
deg_130_compton_df = pd.DataFrame(data = [deg_130_compton], columns = ['130'])

deg_140_counts = (deg_140_v1_300s_df['Events_N'] + deg_140_v2_300s_df['Events_N'] + deg_140_v3_300s_df['Events_N']) / 3
deg_140_compton = deg_140_counts - deg_140_background_300s_df['Events_N']
deg_140_compton_df = pd.DataFrame(data = [deg_140_compton], columns = ['140'])


#%%

#compton_arrays_df = pd.concat([deg_20_compton_df, deg_30_compton_df, deg_45_compton_df, deg_50_compton_df, deg_60_compton_df, deg_70_compton_df, deg_80_compton_df, deg_90_compton_df, deg_100_compton_df, deg_110_compton_df, deg_120_compton_df, deg_130_compton_df, deg_140_compton_df], axis = 1,  keys=[ '20', '30', '45', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140']) 

compton_arrays_df = pd.concat([deg_20_compton, deg_30_compton, deg_45_compton, deg_50_compton, deg_60_compton, deg_70_compton, deg_80_compton, deg_90_compton, deg_100_compton, deg_110_compton, deg_120_compton, deg_130_compton, deg_140_compton], axis = 1,  keys=[ '20', '30', '45', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140']) 


channels = deg_45_background_df['channel_n']


#%%

calibrated_energies = np.loadtxt('calibrated_energies_non_and_lin.csv')

autocalibrated_energies = deg_45_background_df['Energy_keV']
 

angles_deg = np.array((20, 30, 45, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140))

angles_deg_uncerts = np.array(())

# angle uncert is ±5.28°
# need to propagate this through to the expectation values? uncertainty in expectation
# can't really propagate to measured values
# Then look at chi squared tests

angles_rad = angles_deg * np.pi / 180

expected_vals = equation_1(angles_rad)


energies_compton = []
uncerts_compton = []

for i in np.arange(0, len(angles_rad)):
    
    angle = angles_deg[i]
    
    angle_rad = angles_rad[i]
    
    print('angle degrees')
    print(angle)
    
    expect  = equation_1(angle_rad)
    
    print('expectation value')
    print(expect)
    
    channel_val = (expect - 23.35650629722379) / 1.6445278681687145
    
    
    print('channel value')
    print(channel_val)
    
    if angle < 100:
    
        energies_fitting_range = calibrated_energies[int(channel_val - 25): int(channel_val + 60)]
    
        energies_fitting_range_df = pd.DataFrame(energies_fitting_range)
    
        print('energies fitting range')
        print(energies_fitting_range)
    
        compton_array = compton_arrays_df[str(angle)] + 7
    
    
        print('comtpon array')
        print(compton_array)
    
        compton_fitting_range_df = compton_array[int(channel_val - 25): int(channel_val + 60)]
    
        compton_fitting_range = compton_fitting_range_df.values.flatten()
    
        print('compton fitting range')
        print(compton_fitting_range)
    
        x = energies_fitting_range

        y = compton_fitting_range
    
        plt.plot(calibrated_energies, compton_array, linewidth = 0, marker = '.')
    
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
        plt.grid()
        plt.savefig(str(angle) + 'Compton_peak_fit')
        plt.show()
    
        uncerts = (np.sqrt(np.diag(pcov)))
        energy_uncert = uncerts[1]
        energy = popt[1]

        energies_compton.append(energy)
        uncerts_compton.append(energy_uncert)
        
    elif angle < 140:
        
        energies_fitting_range = calibrated_energies[int(channel_val): int(channel_val + 75)]
    
        energies_fitting_range_df = pd.DataFrame(energies_fitting_range)
    
        print('energies fitting range')
        print(energies_fitting_range)
    
        compton_array = compton_arrays_df[str(angle)] + 7
    
    
        print('comtpon array')
        print(compton_array)
    
        compton_fitting_range_df = compton_array[int(channel_val): int(channel_val + 75)]
    
        compton_fitting_range = compton_fitting_range_df.values.flatten()
    
        print('compton fitting range')
        print(compton_fitting_range)
    
        x = energies_fitting_range

        y = compton_fitting_range
    
        plt.plot(calibrated_energies, compton_array, linewidth = 0, marker = '.')
    
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
        plt.grid()
        plt.savefig(str(angle) + 'Compton_peak_fit')
        plt.show()
    
        uncerts = (np.sqrt(np.diag(pcov)))
        energy_uncert = uncerts[1]
        energy = popt[1]

        energies_compton.append(energy)
        uncerts_compton.append(energy_uncert)
        
        
    else:
        
        energies_fitting_range = calibrated_energies[int(channel_val) + 5: int(channel_val + 45)]
    
        energies_fitting_range_df = pd.DataFrame(energies_fitting_range)
    
        print('energies fitting range')
        print(energies_fitting_range)
    
        compton_array = compton_arrays_df[str(angle)] + 7
    
    
        print('comtpon array')
        print(compton_array)
    
        compton_fitting_range_df = compton_array[int(channel_val) + 5 : int(channel_val + 45)]
    
        compton_fitting_range = compton_fitting_range_df.values.flatten()
    
        print('compton fitting range')
        print(compton_fitting_range)
    
        x = energies_fitting_range

        y = compton_fitting_range
    
        plt.plot(calibrated_energies, compton_array, linewidth = 0, marker = '.')
    
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
        plt.grid()
        plt.savefig(str(angle) + 'Compton_peak_fit')
        plt.show()
    
        uncerts = (np.sqrt(np.diag(pcov)))
        energy_uncert = uncerts[1]
        energy = popt[1]

        energies_compton.append(energy)
        uncerts_compton.append(energy_uncert)
        
        
    
#%%

# Add in error bars for angles

plt.errorbar(angles_deg, energies_compton, yerr = uncerts_compton, label = 'measured')
plt.plot(angles_deg, expected_vals, label = 'expected')
plt.title('Compton Effect Verification')
plt.grid()
plt.legend()
plt.xlabel('Angle (degrees)')
plt.ylabel('Energy of Scattered Photon (keV)')
plt.savefig('Compton Effect Verification')
plt.show()

#%%

# 300s measurement series 

# Plots for 45 degree example



plt.plot(calibrated_energies, deg_45_background_df['Events_N'], marker = 'x', linewidth = 0, label = 'Background')
plt.xlabel('Energy (keV)')
plt.ylabel('Counts')

plt.plot(calibrated_energies, deg_45_counts, marker = '.', linewidth = 0, label = 'Scattered')
plt.xlabel('Energy (keV)')
plt.ylabel('Counts')
plt.grid()
plt.title('Background and Scattered 45 degrees')
plt.legend()
plt.savefig('Background and Scattered 45 degrees')
plt.show()


plt.plot(calibrated_energies, deg_45_compton, marker = '.', linewidth = 0)
plt.xlabel('Energy (keV)')
plt.ylabel('Counts')
plt.title('Compton 45 Degrees')
plt.grid()
plt.savefig('Compton 45 Degrees')
plt.show()



#%%











