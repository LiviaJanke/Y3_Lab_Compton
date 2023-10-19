# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:10:01 2023

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

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


# first part with least squares
from scipy.optimize import curve_fit

# second part about ODR
from scipy.odr import ODR, Model, Data, RealData

# style and notebook integration of the plots
import seaborn as sns


#%%

# import data

# AM source
# peak between 40 and 60
Am_241_625V_v1_df = pd.read_csv('Calibration_data_files/Am__241_625V_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])
Am_241_625V_v2_df = pd.read_csv('Calibration_data_files/Am__241_625V_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])
Am_241_625V_v3_df = pd.read_csv('Calibration_data_files/Am__241_625V_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])


#Cs Source
# peak between [385:450]
Cs_137_625V_df = pd.read_csv('Calibration_data_files/Cs_137_625V.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])
Cs_137_625V_v2_df = pd.read_csv('Calibration_data_files/Cs_137_625V_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])
Cs_137_625V_v3_df = pd.read_csv('Calibration_data_files/Cs_137_625V_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])

#Co Source
# [84:107]
Co_57_625V_v1_df = pd.read_csv('Calibration_data_files/Co_57_625V_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])
Co_57_625V_v2_df = pd.read_csv('Calibration_data_files/Co_57_625V_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])
Co_57_625V_v3_df = pd.read_csv('Calibration_data_files/Co_57_625V_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])

#BA133 
# peak between [0:50], [50:100], [220:300]

Ba_133_625V_v1_df = pd.read_csv('Calibration_data_files/Ba__133_625V_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])
Ba_133_625V_v2_df = pd.read_csv('Calibration_data_files/Ba__133_625V_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])
Ba_133_625V_v3_df = pd.read_csv('Calibration_data_files/Ba__133_625V_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])

#Na 22

Na_22_625V_v1_df = pd.read_csv('Calibration_data_files/Na__22_625V_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])
Na_22_625V_v2_df = pd.read_csv('Calibration_data_files/Na__22_625V_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])
Na_22_625V_v3_df = pd.read_csv('Calibration_data_files/Na__22_625V_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])

#%%


# create peak fitting function

def Gaussian_peak_fit(xarray, yarray):
    
    x = xarray
    
    y = yarray
    
    # weighted arithmetic mean
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

    
    popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

    plt.plot(x, y, 'b+:', label='data')
    plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
    plt.legend()
    plt.title('Fit')
    plt.xlabel('Channel')
    plt.ylabel('Counts')
    plt.show()


    uncerts = (np.sqrt(np.diag(pcov)))

    channel_no = popt[1]
    
    channel_no_uncert = uncerts[1]
    
    return channel_no, channel_no_uncert, sigma



#%%

# Applying peak fit to every dataset


# Am
Am_v1_channel_no, Am_v1_channel_no_uncert, Am_v1_sigma = Gaussian_peak_fit(Am_241_625V_v1_df['channel_n'][40:60], Am_241_625V_v1_df['Events_N'][40:60])
Am_v2_channel_no, Am_v2_channel_no_uncert, Am_v2_sigma = Gaussian_peak_fit(Am_241_625V_v2_df['channel_n'][40:60], Am_241_625V_v2_df['Events_N'][40:60])
Am_v3_channel_no, Am_v3_channel_no_uncert, Am_v3_sigma = Gaussian_peak_fit(Am_241_625V_v3_df['channel_n'][40:60], Am_241_625V_v3_df['Events_N'][40:60])


# Cs

Cs_v1_channel_no, Cs_v1_channel_no_uncert, Cs_v1_sigma = Gaussian_peak_fit(Cs_137_625V_df['channel_n'][385:450], Cs_137_625V_df['Events_N'][385:450])
Cs_v2_channel_no, Cs_v2_channel_no_uncert, Cs_v2_sigma = Gaussian_peak_fit(Cs_137_625V_v2_df['channel_n'][385:450], Cs_137_625V_v2_df['Events_N'][385:450])
Cs_v3_channel_no, Cs_v3_channel_no_uncert, Cs_v3_sigma = Gaussian_peak_fit(Cs_137_625V_v3_df['channel_n'][385:450], Cs_137_625V_v3_df['Events_N'][385:450])

# Co

Co_v1_channel_no, Co_v1_channel_no_uncert, Co_v1_sigma = Gaussian_peak_fit(Co_57_625V_v1_df['channel_n'][84:107], Co_57_625V_v1_df['Events_N'][84:107])
Co_v2_channel_no, Co_v2_channel_no_uncert, Co_v2_sigma = Gaussian_peak_fit(Co_57_625V_v2_df['channel_n'][84:107], Co_57_625V_v2_df['Events_N'][84:107])
Co_v3_channel_no, Co_v3_channel_no_uncert, Co_v3_sigma = Gaussian_peak_fit(Co_57_625V_v3_df['channel_n'][84:107], Co_57_625V_v3_df['Events_N'][84:107])

# Ba p1

Ba_v1_channel_no_p1, Ba_v1_channel_no_uncert_p1, Ba_v1_sigma_p1 = Gaussian_peak_fit(Ba_133_625V_v1_df['channel_n'][0:50], Ba_133_625V_v1_df['Events_N'][0:50])
Ba_v2_channel_no_p1, Ba_v2_channel_no_uncert_p1, Ba_v2_sigma_p1 = Gaussian_peak_fit(Ba_133_625V_v2_df['channel_n'][0:50], Ba_133_625V_v2_df['Events_N'][0:50])
Ba_v3_channel_no_p1, Ba_v3_channel_no_uncert_p1, Ba_v3_sigma_p1 = Gaussian_peak_fit(Ba_133_625V_v3_df['channel_n'][0:50], Ba_133_625V_v3_df['Events_N'][0:50])

# Ba p2

Ba_v1_channel_no_p2, Ba_v1_channel_no_uncert_p2, Ba_v1_sigma_p2 = Gaussian_peak_fit(Ba_133_625V_v1_df['channel_n'][50:100], Ba_133_625V_v1_df['Events_N'][50:100])
Ba_v2_channel_no_p2, Ba_v2_channel_no_uncert_p2, Ba_v2_sigma_p2 = Gaussian_peak_fit(Ba_133_625V_v2_df['channel_n'][50:100], Ba_133_625V_v2_df['Events_N'][50:100])
Ba_v3_channel_no_p2, Ba_v3_channel_no_uncert_p2, Ba_v3_sigma_p2 = Gaussian_peak_fit(Ba_133_625V_v3_df['channel_n'][50:100], Ba_133_625V_v3_df['Events_N'][50:100])


#%%
























