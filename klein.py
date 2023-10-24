# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:31:47 2023

@author: baidi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as math
import matplotlib.dates as mdates
from scipy.optimize import curve_fit

from sklearn.linear_model import LinearRegression
from scipy.stats import linregress

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


# first part with least squares
from scipy.optimize import curve_fit

# second part about ODR
from scipy.odr import ODR, Model, Data, RealData

# style and notebook integration of the plots
import seaborn as sns
#%%
background = pd.read_csv('Angles_200s_meas/0_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
plt.plot(background['channel_n'], background['Events_N'])
#%% 10deg
deg10_v1 = pd.read_csv('Angles_200s_meas/10_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg10_v2 = pd.read_csv('Angles_200s_meas/10_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg10_v3 = pd.read_csv('Angles_200s_meas/10_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg10_background = pd.read_csv('Angles_200s_meas/10_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
plt.plot(deg10_v1['channel_n'], deg10_v1['Events_N'])
#%% 20 deg
deg20_v1 = pd.read_csv('Angles_200s_meas/20_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg20_v2 = pd.read_csv('Angles_200s_meas/20_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg20_v3 = pd.read_csv('Angles_200s_meas/20_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg20_background = pd.read_csv('Angles_200s_meas/20_deg_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
plt.plot(deg20_v1['channel_n'], deg20_v1['Events_N'])
#%% 30 deg
deg30_v1 = pd.read_csv('Angles_200s_meas/30_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg30_v2 = pd.read_csv('Angles_200s_meas/30_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg30_v3 = pd.read_csv('Angles_200s_meas/30_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg30_background = pd.read_csv('Angles_200s_meas/30_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
#%% 45 deg
deg45_v1 = pd.read_csv('Angles_200s_meas/45_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg45_v2 = pd.read_csv('Angles_200s_meas/45_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg45_v3 = pd.read_csv('Angles_200s_meas/45_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg45_background = pd.read_csv('Angles_200s_meas/45_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
#%% 50 deg
#deg50_v1 = pd.read_csv('Angles_200s_meas/50_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg50_v2 = pd.read_csv('Angles_200s_meas/50_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg50_v3 = pd.read_csv('Angles_200s_meas/50_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg50_background = pd.read_csv('Angles_200s_meas/50_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
#%% 60 deg
deg60_v1 = pd.read_csv('Angles_200s_meas/60_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg60_v2 = pd.read_csv('Angles_200s_meas/60_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg60_v3 = pd.read_csv('Angles_200s_meas/60_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg60_background = pd.read_csv('Angles_200s_meas/60_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
#%% 70 deg
deg70_v1 = pd.read_csv('Angles_200s_meas/70_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg70_v2 = pd.read_csv('Angles_200s_meas/70_deg_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg70_v3 = pd.read_csv('Angles_200s_meas/70_deg_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg70_background = pd.read_csv('Angles_200s_meas/70_deg_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
#%% 80 deg
deg80_v1 = pd.read_csv('Angles_200s_meas/80_deg_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg80_v2 = pd.read_csv('Angles_200s_meas/80_deg_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg80_v3 = pd.read_csv('Angles_200s_meas/80_deg_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg80_background = pd.read_csv('Angles_200s_meas/80_deg_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
#%% 90 deg
deg90_v1 = pd.read_csv('Angles_200s_meas/90_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg90_v2 = pd.read_csv('Angles_200s_meas/90_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg90_v3 = pd.read_csv('Angles_200s_meas/90_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg90_background = pd.read_csv('Angles_200s_meas/90_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
#%% 100 deg
deg100_v1 = pd.read_csv('Angles_200s_meas/100_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg100_v2 = pd.read_csv('Angles_200s_meas/100_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg100_v3 = pd.read_csv('Angles_200s_meas/100_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg100_background = pd.read_csv('Angles_200s_meas/100_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
#%% 110 deg
deg110_v1 = pd.read_csv('Angles_200s_meas/110_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg110_v2 = pd.read_csv('Angles_200s_meas/110_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg110_v3 = pd.read_csv('Angles_200s_meas/110_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg110_background = pd.read_csv('Angles_200s_meas/110_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
#%% 120 deg
deg120_v1 = pd.read_csv('Angles_200s_meas/120_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg120_v2 = pd.read_csv('Angles_200s_meas/120_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg120_v3 = pd.read_csv('Angles_200s_meas/120_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg120_background = pd.read_csv('Angles_200s_meas/120_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
#%% 130 deg
deg130_v1 = pd.read_csv('Angles_200s_meas/130_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg130_v2 = pd.read_csv('Angles_200s_meas/130_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg130_v3 = pd.read_csv('Angles_200s_meas/130_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg130_background = pd.read_csv('Angles_200s_meas/130_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
#%% 140 deg
deg140_v1 = pd.read_csv('Angles_200s_meas/140_deg_300s_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg140_v2 = pd.read_csv('Angles_200s_meas/140_deg_300s_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg140_v3 = pd.read_csv('Angles_200s_meas/140_deg_300s_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
deg140_background = pd.read_csv('Angles_200s_meas/140_deg_300s_background.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
#%% 
deg10 = [np.nansum(deg10_v1['rate_r_1/S']), np.nansum(deg10_v2['rate_r_1/S']), np.nansum(deg10_v3['rate_r_1/S'])]
deg10_count = (np.nansum(deg10_v1['rate_r_1/S'])+np.nansum(deg10_v2['rate_r_1/S'])+np.nansum(deg10_v3['rate_r_1/S']))/3
deg10_count_back = (np.nansum(deg10_background['rate_r_1/S']))
deg10_mean_count = (deg10_count)/300
deg10_compton_mean_count = (deg10_count - deg10_count_back)/300
deg10uncert = (max(deg10)-min(deg10))/300/2
#%%
deg20 = [np.nansum(deg20_v1['rate_r_1/S']), np.nansum(deg20_v2['rate_r_1/S']), np.nansum(deg20_v3['rate_r_1/S'])]
deg20_count = (np.nansum(deg20_v1['rate_r_1/S'])+np.nansum(deg20_v2['rate_r_1/S'])+np.nansum(deg20_v3['rate_r_1/S']))/3
deg20_count_back = (np.nansum(deg20_background['rate_r_1/S']))
deg20_mean_count = (deg20_count)/300
deg20_compton_mean_count = (deg20_count - deg20_count_back)/300
deg20uncert = (max(deg20)-min(deg20))/300/2
#%%
deg30 = [np.nansum(deg30_v1['rate_r_1/S']), np.nansum(deg30_v2['rate_r_1/S']), np.nansum(deg30_v3['rate_r_1/S'])]
deg30_count = (np.nansum(deg30_v1['rate_r_1/S'])+np.nansum(deg30_v2['rate_r_1/S'])+np.nansum(deg30_v3['rate_r_1/S']))/3
deg30_count_back = (np.nansum(deg30_background['rate_r_1/S']))
deg30_mean_count = (deg30_count)/300
deg30_compton_mean_count = (deg30_count - deg30_count_back)/300
deg30uncert = (max(deg30)-min(deg30))/300/2
#%%
deg45 = [np.nansum(deg45_v1['rate_r_1/S']), np.nansum(deg45_v2['rate_r_1/S']), np.nansum(deg45_v3['rate_r_1/S'])]
deg45_count = (np.nansum(deg45_v1['rate_r_1/S'])+np.nansum(deg45_v2['rate_r_1/S'])+np.nansum(deg45_v3['rate_r_1/S']))/3
deg45_count_back = (np.nansum(deg45_background['rate_r_1/S']))
deg45_mean_count = (deg45_count)/300
deg45_compton_mean_count = (deg45_count - deg45_count_back)/300
deg45uncert = (max(deg45)-min(deg45))/300/2
#%%
deg50 = [np.nansum(deg50_v2['rate_r_1/S']), np.nansum(deg50_v3['rate_r_1/S'])]
deg50_count = (np.nansum(deg50_v2['rate_r_1/S'])+np.nansum(deg50_v3['rate_r_1/S']))/2
deg50_count_back = (np.nansum(deg50_background['rate_r_1/S']))
deg50_mean_count = (deg50_count)/300
deg50_compton_mean_count = (deg50_count - deg50_count_back)/300
deg50uncert = (max(deg50)-min(deg50))/300/2
#%%
deg60 = [np.nansum(deg60_v1['rate_r_1/S']), np.nansum(deg60_v2['rate_r_1/S']), np.nansum(deg60_v3['rate_r_1/S'])]
deg60_count = (np.nansum(deg60_v1['rate_r_1/S'])+np.nansum(deg60_v2['rate_r_1/S'])+np.nansum(deg60_v3['rate_r_1/S']))/3
deg60_count_back = (np.nansum(deg60_background['rate_r_1/S']))
deg60_mean_count = (deg60_count)/300
deg60_compton_mean_count = (deg60_count - deg60_count_back)/300
deg60uncert = (max(deg60)-min(deg60))/300/2
#%%
deg70 = [np.nansum(deg70_v1['rate_r_1/S']), np.nansum(deg70_v2['rate_r_1/S']), np.nansum(deg70_v3['rate_r_1/S'])]
deg70_count = (np.nansum(deg70_v1['rate_r_1/S'])+np.nansum(deg70_v2['rate_r_1/S'])+np.nansum(deg70_v3['rate_r_1/S']))/3
deg70_count_back = (np.nansum(deg70_background['rate_r_1/S']))
deg70_mean_count = (deg70_count)/200
deg70_compton_mean_count = (deg70_count - deg70_count_back)/200
deg70uncert = (max(deg70)-min(deg70))/200/2
#%%
deg80 = [np.nansum(deg80_v1['rate_r_1/S']), np.nansum(deg80_v2['rate_r_1/S']), np.nansum(deg80_v3['rate_r_1/S'])]
deg80_count = (np.nansum(deg80_v1['rate_r_1/S'])+np.nansum(deg80_v2['rate_r_1/S'])+np.nansum(deg80_v3['rate_r_1/S']))/3
deg80_count_back = (np.nansum(deg80_background['rate_r_1/S']))
deg80_mean_count = (deg80_count)/200
deg80_compton_mean_count = (deg80_count - deg80_count_back)/200
deg80uncert = (max(deg80)-min(deg80))/200/2
#%%
deg90 = [np.nansum(deg90_v1['rate_r_1/S']), np.nansum(deg90_v2['rate_r_1/S']), np.nansum(deg90_v3['rate_r_1/S'])]
deg90_count = (np.nansum(deg90_v1['rate_r_1/S'])+np.nansum(deg90_v2['rate_r_1/S'])+np.nansum(deg90_v3['rate_r_1/S']))/3
deg90_count_back = (np.nansum(deg90_background['rate_r_1/S']))
deg90_mean_count = (deg90_count)/300
deg90_compton_mean_count = (deg90_count - deg90_count_back)/300
deg90uncert = (max(deg90)-min(deg90))/300/2
#%%
deg100 = [np.nansum(deg100_v1['rate_r_1/S']), np.nansum(deg100_v2['rate_r_1/S']), np.nansum(deg100_v3['rate_r_1/S'])]
deg100_count = (np.nansum(deg100_v1['rate_r_1/S'])+np.nansum(deg100_v2['rate_r_1/S'])+np.nansum(deg100_v3['rate_r_1/S']))/3
deg100_count_back = (np.nansum(deg100_background['rate_r_1/S']))
deg100_mean_count = (deg100_count)/300
deg100_compton_mean_count = (deg100_count - deg100_count_back)/300
deg100uncert = (max(deg100)-min(deg100))/300/2
#%%
deg110 = [np.nansum(deg110_v1['rate_r_1/S']), np.nansum(deg110_v2['rate_r_1/S']), np.nansum(deg110_v3['rate_r_1/S'])]
deg110_count = (np.nansum(deg110_v1['rate_r_1/S'])+np.nansum(deg110_v2['rate_r_1/S'])+np.nansum(deg110_v3['rate_r_1/S']))/3
deg110_count_back = (np.nansum(deg110_background['rate_r_1/S']))
deg110_mean_count = (deg110_count)/300
deg110_compton_mean_count = (deg110_count - deg110_count_back)/300
deg110uncert = (max(deg110)-min(deg110))/300/2
#%%
deg120 = [np.nansum(deg10_v1['rate_r_1/S']), np.nansum(deg120_v2['rate_r_1/S']), np.nansum(deg120_v3['rate_r_1/S'])]
deg120_count = (np.nansum(deg120_v1['rate_r_1/S'])+np.nansum(deg120_v2['rate_r_1/S'])+np.nansum(deg120_v3['rate_r_1/S']))/3
deg120_count_back = (np.nansum(deg120_background['rate_r_1/S']))
deg120_mean_count = (deg120_count)/300
deg120_compton_mean_count = (deg120_count - deg120_count_back)/300
deg120uncert = (max(deg120)-min(deg120))/300/2
#%%
deg130 = [np.nansum(deg130_v1['rate_r_1/S']), np.nansum(deg130_v2['rate_r_1/S']), np.nansum(deg130_v3['rate_r_1/S'])]
deg130_count = (np.nansum(deg130_v1['rate_r_1/S'])+np.nansum(deg130_v2['rate_r_1/S'])+np.nansum(deg130_v3['rate_r_1/S']))/3
deg130_count_back = (np.nansum(deg130_background['rate_r_1/S']))
deg130_mean_count = (deg130_count)/300
deg130_compton_mean_count = (deg130_count - deg130_count_back)/300
deg130uncert = (max(deg130)-min(deg130))/300/2
#%%
deg140 = [np.nansum(deg140_v1['rate_r_1/S']), np.nansum(deg140_v2['rate_r_1/S']), np.nansum(deg140_v3['rate_r_1/S'])]
deg140_count = (np.nansum(deg140_v1['rate_r_1/S'])+np.nansum(deg140_v2['rate_r_1/S'])+np.nansum(deg140_v3['rate_r_1/S']))/3
deg140_count_back = (np.nansum(deg140_background['rate_r_1/S']))
deg140_mean_count = (deg140_count)/300
deg140_compton_mean_count = (deg140_count - deg140_count_back)/300
deg140uncert = (max(deg140)-min(deg140))/300/2
#%%
def expklein(y):
    return (y/(0.22))/(9.8*10**-2*1.1168*10**25*5059.0875) #4047.27

degmeancount = []
degexpklein = []
degexpkleinuncert = []

deg10_expklein = expklein(deg10_mean_count)
#degmeancount.append(deg10_mean_count)
#degexpklein.append(deg10_expklein)
#degexpkleinuncert.append(deg10uncert)
deg20_expklein = expklein(deg20_mean_count)
degmeancount.append(deg20_mean_count)
degexpklein.append(deg20_expklein)
degexpkleinuncert.append(deg20uncert)
deg30_expklein = expklein(deg30_mean_count)
degmeancount.append(deg30_mean_count)
degexpklein.append(deg30_expklein)
degexpkleinuncert.append(deg30uncert)
deg45_expklein = expklein(deg45_mean_count)
degmeancount.append(deg45_mean_count)
degexpklein.append(deg45_expklein)
degexpkleinuncert.append(deg45uncert)
deg50_expklein = expklein(deg50_mean_count)
degmeancount.append(deg50_mean_count)
degexpklein.append(deg50_expklein)
degexpkleinuncert.append(deg50uncert)
deg60_expklein = expklein(deg60_mean_count)
degmeancount.append(deg60_mean_count)
degexpklein.append(deg60_expklein)
degexpkleinuncert.append(deg60uncert)
deg70_expklein = expklein(deg70_mean_count)
degmeancount.append(deg70_mean_count)
degexpklein.append(deg70_expklein)
degexpkleinuncert.append(deg70uncert)
deg80_expklein = expklein(deg80_mean_count)
degmeancount.append(deg80_mean_count)
degexpklein.append(deg80_expklein)
degexpkleinuncert.append(deg80uncert)
deg90_expklein = expklein(deg90_mean_count)
degmeancount.append(deg90_mean_count)
degexpklein.append(deg90_expklein)
degexpkleinuncert.append(deg90uncert)
deg100_expklein = expklein(deg100_mean_count)
degmeancount.append(deg100_mean_count)
degexpklein.append(deg100_expklein)
degexpkleinuncert.append(deg100uncert)
deg110_expklein = expklein(deg110_mean_count)
degmeancount.append(deg110_mean_count)
degexpklein.append(deg110_expklein)
degexpkleinuncert.append(deg110uncert)
deg120_expklein = expklein(deg120_mean_count)
#degmeancount.append(deg120_mean_count)
#degexpklein.append(deg120_expklein)
#degexpkleinuncert.append(deg120uncert)
deg130_expklein = expklein(deg130_mean_count)
#degmeancount.append(deg130_mean_count)
#degexpklein.append(deg130_expklein)
#degexpkleinuncert.append(deg130uncert)
deg140_expklein = expklein(deg140_mean_count)
#degmeancount.append(deg140_mean_count)
#degexpklein.append(deg140_expklein)
#degexpkleinuncert.append(deg140uncert)


#degreesexpklein = [deg10_expklein, deg20_expklein, deg30_expklein, deg45_expklein. deg60_expklein. deg70_expklein, deg80_expklein, deg90_expklein, deg100_expklein, deg110_expklein, deg120_expklein, deg130_expklein, deg140_expklein]
degree = [20,30,45,50,60,70,80,90,100,110]

#Uncertainty of dOmega = 0.1866*10**-26/percentage:0.403
uncertklein = degexpklein*np.sqrt((0.206)**2+(np.array(degexpkleinuncert)/np.array(degmeancount))**2)



val_linear = (np.linspace(0, 150, 500))
def klein(a,g):
    r = a/180*np.pi
    return (2.82*10**-13)**2*((1+(np.cos(r))**2)/2)*(1/(1+g*(1-np.cos(r)))**2)*(1+((g*(1-np.cos(r)))**2/((1+np.cos(r)**2)*(1+g*(1-np.cos(r))))))

plt.plot(val_linear, klein(val_linear,1.29), label = 'Theoretical')
plt.errorbar(degree,degexpklein, xerr = 0, yerr = uncertklein, label = 'data', marker = 's')
#plt.plot(degree, degexpklein, label = 'data', marker = 'x', linewidth = 0)
plt.title('Differential Cross Section Fit')
plt.xlabel('Angle of Scattering/(degrees)')
plt.ylabel('Cross Section/(cm^2*10^-27/steradian)')
plt.grid()
plt.legend()



#%%
def compton(a):
    r = a/180*np.pi
    E_gamma_0 = 661.7
    mec2 = 511
    E_gamma_theta = E_gamma_0 / (1 + (E_gamma_0 / mec2) * (1 - np.cos(r)))
    return E_gamma_theta

effi = []
g = 1.29

effi20 = deg20_expklein/klein(20, g)
effi.append(effi20)
effi30 = deg30_expklein/klein(30, g)
effi.append(effi30)
effi45 = deg45_expklein/klein(45, g)
effi.append(effi45)
effi50 = deg50_expklein/klein(50, g)
effi.append(effi50)
effi60 = deg60_expklein/klein(60, g)
effi.append(effi60)
effi70 = deg70_expklein/klein(70, g)
effi.append(effi70)
effi80 = deg80_expklein/klein(80, g)
effi.append(effi80)
effi90 = deg90_expklein/klein(90, g)
effi.append(effi90)
effi100 = deg100_expklein/klein(100, g)
effi.append(effi100)
effi110 = deg110_expklein/klein(110, g)
effi.append(effi110)
effi120 = deg120_expklein/klein(120, g)
#effi.append(effi120)
effi130 = deg130_expklein/klein(130, g)
#effi.append(effi130)

ecompton = compton(np.array(degree))
uncerteffi = np.array(degexpkleinuncert)*effi

#plt.plot(degree, effi, label = 'Efficiency', marker = 'o', linewidth = 0)

plt.errorbar(ecompton, effi, xerr= 0, yerr = 0, label = 'Efficiency', marker = 's')
plt.title('Scintillator Detector Experimental Efficiency')
plt.xlabel('Compton Scattering Energy/(kev)')
plt.ylabel('Efficiency')
plt.grid()
plt.legend()
#%%
plt.plot(degree, effi, label = 'Efficiency', marker = 'o', linewidth = 0)
#%%
def thomson(a):
    r = a/180*np.pi
    return (2.82*10**-13)**2*((1+(np.cos(r))**2)/2)

plt.plot(val_linear, thomson(val_linear), label = 'Thomson')
plt.plot(val_linear, klein(val_linear,1.29), label = 'Theoretical')

#%%
def correctklein(y,E):
    return (y/(0.22*E))/(9.8*10**-2*1.1168*10**25*8609.168) #8609.168
degexpklein = []

deg10_expklein = correctklein(deg10_mean_count, 1)
#degexpklein.append(deg10_expklein)
deg20_expklein = correctklein(deg20_mean_count, 0.47)
degexpklein.append(deg20_expklein)
deg30_expklein = correctklein(deg30_mean_count, 0.49)
degexpklein.append(deg30_expklein)
deg45_expklein = correctklein(deg45_mean_count, 0.52)
degexpklein.append(deg45_expklein)
deg50_expklein = correctklein(deg50_mean_count, 0.55)
degexpklein.append(deg50_expklein)
deg60_expklein = correctklein(deg60_mean_count, 0.60)
degexpklein.append(deg60_expklein)
deg70_expklein = correctklein(deg70_mean_count, 0.65)
degexpklein.append(deg70_expklein)
deg80_expklein = correctklein(deg80_mean_count, 0.69)
degexpklein.append(deg80_expklein)
deg90_expklein = correctklein(deg90_mean_count, 0.72)
degexpklein.append(deg90_expklein)
deg100_expklein = correctklein(deg100_mean_count, 0.75)
degexpklein.append(deg100_expklein)
deg110_expklein = correctklein(deg110_mean_count, 0.78)
degexpklein.append(deg110_expklein)
deg120_expklein = correctklein(deg120_mean_count, 0.81)
#degexpklein.append(deg120_expklein)
deg130_expklein = correctklein(deg130_mean_count, 0.84)
#degexpklein.append(deg130_expklein)
deg140_expklein = correctklein(deg140_mean_count, 0.86)
#degexpklein.append(deg140_expklein)


#degreesexpklein = [deg10_expklein, deg20_expklein, deg30_expklein, deg45_expklein. deg60_expklein. deg70_expklein, deg80_expklein, deg90_expklein, deg100_expklein, deg110_expklein, deg120_expklein, deg130_expklein, deg140_expklein]
degree = [20,30,45,50,60,70,80,90,100,110]



val_linear = (np.linspace(0, 150, 500))
def klein(a,g):
    r = a/180*np.pi
    return (2.82*10**-13)**2*((1+(np.cos(r))**2)/2)*(1/(1+g*(1-np.cos(r)))**2)*(1+((g*(1-np.cos(r)))**2/((1+np.cos(r)**2)*(1+g*(1-np.cos(r))))))


plt.plot(val_linear, klein(val_linear,1.29), label = 'Theoretical', marker = 'o')
plt.errorbar(degree, degexpklein, xerr = 0, yerr = uncertklein, label = 'data', marker = 's')
#plt.plot(degree, degexpklein, label = 'data', marker = 'x', linewidth = 0)
plt.title('Differential Cross Section Fit')
plt.xlabel('Angle of Scattering/(degrees)')
plt.ylabel('Cross Section/(cm^2*10^-27/steradian)')
plt.legend()

#%%
def correctklein(y,E):
    return (y/(0.22*E))/(9.8*10**-2*1.1168*10**25*8609.168) #8609.168
degexpklein = []

deg10_expklein = correctklein(deg10_mean_count, 1)
#degexpklein.append(deg10_expklein)
deg20_expklein = correctklein(deg20_mean_count, 0.18)
degexpklein.append(deg20_expklein)
deg30_expklein = correctklein(deg30_mean_count, 0.20)
degexpklein.append(deg30_expklein)
deg45_expklein = correctklein(deg45_mean_count, 0.28)
degexpklein.append(deg45_expklein)
deg50_expklein = correctklein(deg50_mean_count, 0.30)
degexpklein.append(deg50_expklein)
deg60_expklein = correctklein(deg60_mean_count, 0.37)
degexpklein.append(deg60_expklein)
deg70_expklein = correctklein(deg70_mean_count, 0.47)
degexpklein.append(deg70_expklein)
deg80_expklein = correctklein(deg80_mean_count, 0.55)
degexpklein.append(deg80_expklein)
deg90_expklein = correctklein(deg90_mean_count, 0.60)
degexpklein.append(deg90_expklein)
deg100_expklein = correctklein(deg100_mean_count, 0.63)
degexpklein.append(deg100_expklein)
deg110_expklein = correctklein(deg110_mean_count, 0.65)
degexpklein.append(deg110_expklein)
deg120_expklein = correctklein(deg120_mean_count, 0.81)
#degexpklein.append(deg120_expklein)
deg130_expklein = correctklein(deg130_mean_count, 0.84)
#degexpklein.append(deg130_expklein)
deg140_expklein = correctklein(deg140_mean_count, 0.86)
#degexpklein.append(deg140_expklein)


#degreesexpklein = [deg10_expklein, deg20_expklein, deg30_expklein, deg45_expklein. deg60_expklein. deg70_expklein, deg80_expklein, deg90_expklein, deg100_expklein, deg110_expklein, deg120_expklein, deg130_expklein, deg140_expklein]
degree = [20,30,45,50,60,70,80,90,100,110]



val_linear = (np.linspace(0, 150, 500))
def klein(a,g):
    r = a/180*np.pi
    return (2.82*10**-13)**2*((1+(np.cos(r))**2)/2)*(1/(1+g*(1-np.cos(r)))**2)*(1+((g*(1-np.cos(r)))**2/((1+np.cos(r)**2)*(1+g*(1-np.cos(r))))))


plt.plot(val_linear, klein(val_linear,1.29), label = 'Theoretical', marker = 'o')
plt.errorbar(degree, degexpklein, xerr = 0, yerr = uncertklein, label = 'data', marker = 's')
#plt.plot(degree, degexpklein, label = 'data', marker = 'x', linewidth = 0)
plt.title('Differential Cross Section Fit')
plt.xlabel('Angle of Scattering/(degrees)')
plt.ylabel('Cross Section/(cm^2*10^-27/steradian)')
plt.legend()
#%%
myeffi = [0.18, 0.20, 0.28, 0.30, 0.37, 0.47, 0.55, 0.60, 0.63, 0.65]
plt.plot(ecompton, myeffi)
#%%
# efficiency curve

def effieq(mu, d):
    return 1-np.exp(-3.67*mu*d)

#murho = [0.081, 0.086, 0.094, 0.104, 0.117, 0.133, 0.152, 0.175, 0.202, 0.232, 0.263, 0.295, 0.324, 0.350]
murho = np.array([0.081, 0.086, 0.094, 0.104, 0.117, 0.133, 0.152, 0.175, 0.202, 0.232])
plt.plot(ecompton, effieq(murho,5.1))
#%%
corrected = degexpklein/effieq(murho,5.1)

plt.plot(val_linear, klein(val_linear,1.29), label = 'Theoretical')
plt.errorbar(degree, corrected, xerr = 0, yerr = uncertklein, marker = 's')
plt.grid()
#%%
effithistime = corrected/klein(np.array(degree),1.29)
plt.plot(ecompton, effithistime)


































