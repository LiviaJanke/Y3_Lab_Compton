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

# Na 22
# peak between [290:350]

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

Ba_v1_channel_no_p1, Ba_v1_channel_no_uncert_p1, Ba_v1_sigma_p1 = Gaussian_peak_fit(Ba_133_625V_v1_df['channel_n'][20:40], Ba_133_625V_v1_df['Events_N'][20:40])
Ba_v2_channel_no_p1, Ba_v2_channel_no_uncert_p1, Ba_v2_sigma_p1 = Gaussian_peak_fit(Ba_133_625V_v2_df['channel_n'][20:40], Ba_133_625V_v2_df['Events_N'][20:40])
Ba_v3_channel_no_p1, Ba_v3_channel_no_uncert_p1, Ba_v3_sigma_p1 = Gaussian_peak_fit(Ba_133_625V_v3_df['channel_n'][20:40], Ba_133_625V_v3_df['Events_N'][20:40])

# Ba p2

Ba_v1_channel_no_p2, Ba_v1_channel_no_uncert_p2, Ba_v1_sigma_p2 = Gaussian_peak_fit(Ba_133_625V_v1_df['channel_n'][50:80], Ba_133_625V_v1_df['Events_N'][50:80])
Ba_v2_channel_no_p2, Ba_v2_channel_no_uncert_p2, Ba_v2_sigma_p2 = Gaussian_peak_fit(Ba_133_625V_v2_df['channel_n'][50:80], Ba_133_625V_v2_df['Events_N'][50:80])
Ba_v3_channel_no_p2, Ba_v3_channel_no_uncert_p2, Ba_v3_sigma_p2 = Gaussian_peak_fit(Ba_133_625V_v3_df['channel_n'][50:80], Ba_133_625V_v3_df['Events_N'][50:80])

# Ba p3

Ba_v1_channel_no_p3, Ba_v1_channel_no_uncert_p3, Ba_v1_sigma_p3 = Gaussian_peak_fit(Ba_133_625V_v1_df['channel_n'][220:260], Ba_133_625V_v1_df['Events_N'][220:260])
Ba_v2_channel_no_p3, Ba_v2_channel_no_uncert_p3, Ba_v2_sigma_p3 = Gaussian_peak_fit(Ba_133_625V_v2_df['channel_n'][220:260], Ba_133_625V_v2_df['Events_N'][220:260])
Ba_v3_channel_no_p3, Ba_v3_channel_no_uncert_p3, Ba_v3_sigma_p3 = Gaussian_peak_fit(Ba_133_625V_v3_df['channel_n'][220:260], Ba_133_625V_v3_df['Events_N'][220:260])

# Na p1

Na_v1_channel_no, Na_v1_channel_no_uncert, Na_v1_sigma = Gaussian_peak_fit(Na_22_625V_v1_df['channel_n'][290:350], Na_22_625V_v1_df['Events_N'][290:350])
Na_v2_channel_no, Na_v2_channel_no_uncert, Na_v2_sigma = Gaussian_peak_fit(Na_22_625V_v2_df['channel_n'][290:350], Na_22_625V_v2_df['Events_N'][290:350])
Na_v3_channel_no, Na_v3_channel_no_uncert, Na_v3_sigma = Gaussian_peak_fit(Na_22_625V_v3_df['channel_n'][290:350], Na_22_625V_v3_df['Events_N'][290:350])


#%%

energy_Am_241 = 59.54
energy_Co_57 = 122
energy_Cs_137 = 662
energy_Ba_133_p1 = 30.85 
energy_Ba_133_p2 = 81.0
energy_Ba_133_p3 = 356.0
energy_Na_22 = 511

mean_channel_num_Am_241 = (Am_v1_channel_no + Am_v2_channel_no + Am_v3_channel_no) / 3

mean_channel_num_Co_57 = (Co_v1_channel_no + Co_v2_channel_no + Co_v3_channel_no) / 3

mean_channel_num_Cs_137 = (Cs_v1_channel_no + Cs_v2_channel_no + Cs_v3_channel_no) / 3

mean_channel_num_Ba_133_p1 = (Ba_v1_channel_no_p1 + Ba_v2_channel_no_p1 + Ba_v3_channel_no_p1) / 3

mean_channel_num_Ba_133_p2 = (Ba_v1_channel_no_p2 + Ba_v2_channel_no_p2 + Ba_v3_channel_no_p2) / 3

mean_channel_num_Ba_133_p3 = (Ba_v1_channel_no_p3 + Ba_v2_channel_no_p3 + Ba_v3_channel_no_p3) / 3

mean_channel_num_Na_22 = (Na_v1_channel_no + Na_v2_channel_no + Na_v3_channel_no) / 3

mean_channel_num_am_241_uncert = (Am_v1_channel_no_uncert + Am_v2_channel_no_uncert + Am_v2_channel_no_uncert) / 3

mean_channel_num_Co_57_uncert = (Co_v1_channel_no_uncert + Co_v2_channel_no_uncert + Co_v3_channel_no_uncert) / 3

mean_channel_num_Cs_137_uncert = (Cs_v1_channel_no_uncert + Cs_v2_channel_no_uncert + Cs_v3_channel_no_uncert) / 3

mean_channel_num_Ba_133_p1_uncert = (Ba_v1_channel_no_uncert_p1 + Ba_v2_channel_no_uncert_p1 + Ba_v3_channel_no_uncert_p1) / 3

mean_channel_num_Ba_133_p2_uncert = (Ba_v1_channel_no_uncert_p2 + Ba_v2_channel_no_uncert_p2 + Ba_v3_channel_no_uncert_p2) / 3

mean_channel_num_Ba_133_p3_uncert = (Ba_v1_channel_no_uncert_p3 + Ba_v2_channel_no_uncert_p3 + Ba_v3_channel_no_uncert_p3) / 3

mean_channel_num_Na_22_uncert = (Na_v1_channel_no_uncert + Na_v2_channel_no_uncert + Na_v3_channel_no_uncert) / 3




#%%

energies = np.array((energy_Am_241, energy_Co_57, energy_Cs_137, energy_Ba_133_p1, energy_Ba_133_p2, energy_Ba_133_p3, energy_Na_22))

channel_nums = np.array((mean_channel_num_Am_241, mean_channel_num_Co_57, mean_channel_num_Cs_137, mean_channel_num_Ba_133_p1, mean_channel_num_Ba_133_p2, mean_channel_num_Ba_133_p3, mean_channel_num_Na_22))

channel_num_uncerts = np.array((mean_channel_num_am_241_uncert, mean_channel_num_Co_57_uncert, mean_channel_num_Cs_137_uncert, mean_channel_num_Ba_133_p1_uncert, mean_channel_num_Ba_133_p2_uncert, mean_channel_num_Ba_133_p3_uncert, mean_channel_num_Na_22_uncert))

print(energies)
print(channel_nums)
print(channel_num_uncerts)

plt.plot(channel_nums,energies, 'o')

#%%

# Extrapolation and fit to calibration data
#fitting energy to channel nums


x_testarray = np.linspace(0, 512, 1000)

xvals = channel_nums
xvals_err = channel_num_uncerts
yvals = energies

p, cov = np.polyfit(xvals,yvals,1, cov = True, w = 1/xvals_err)
m = p[0]
b = p[1]

plt.plot(xvals, yvals, 'yo', x_testarray, m*x_testarray+b, '--k')
plt.show()

scd_ref = -1 * b

vmr = m

uncerts = (np.sqrt(np.diag(cov)))

print('y-intercept is')
print(scd_ref)
print(uncerts[1])
print('grad is')
print(vmr)
print(uncerts[0])


#%%

# fit a curve to find turning point for split betweem linear and nonlinear fit 


def f_model(x, a, c):
    return (a * x) + (c * x**2)

estimate = f_model(x_testarray, 1.6, 1e-9)

plt.plot(x_testarray, estimate, label = 'model', marker = '.', linewidth = 0)
plt.plot(xvals, yvals, label = 'data', marker = 'x', linewidth = 0)
plt.legend()


# running curve fit routine


popt, pcov = curve_fit(
    f=f_model,       # model function
    xdata=xvals,   # x data
    ydata=yvals,   # y data
    p0=(1.6, 1e-9),      # initial value of the parameters
    sigma= xvals_err   # uncertainties on y
)

print(popt)


a_opt, c_opt = popt
print("a = ", a_opt)
print("c = ", c_opt)

perr = np.sqrt(np.diag(pcov))
Da, Dc = perr
print("a = %6.2f +/- %4.2f" % (a_opt, Da))
print("c = %6.2f +/- %4.2f" % (c_opt, Dc))

R2 = np.sum((f_model(xvals, a_opt, c_opt) - yvals.mean())**2) / np.sum((yvals - yvals.mean())**2)
print("r^2 = %10.6f" % R2)


estimates_opt = f_model(x_testarray, a_opt, c_opt)

#%%

plt.plot(x_testarray, estimates_opt, label = 'model', marker = '.', linewidth = 0)
plt.plot(xvals, yvals, label = 'data', marker = 'x', linewidth = 0)
plt.legend()

# Find turning point from curve fit

# Derivative of model func = a + 2cx
# Turning point where a + 2cx = 0 
# x = -a / 2c

turning_point = - a_opt / ( 2 * c_opt)
print(turning_point)

# Take 250 as an estimate for now



#%%

# Splitting calib fit into linear and nonlinear

# Non-linear section 0 to 250


def f_model(x, a, c):
    return (a * x) + (c * x**2)

estimate = f_model(x_testarray, 1.6, 1e-9)

popt, pcov = curve_fit(
    f=f_model,       # model function
    xdata=xvals,   # x data
    ydata=yvals,   # y data
    p0=(1.6, 1e-9),      # initial value of the parameters
    sigma= xvals_err   # uncertainties on y
)

print(popt)


a_opt, c_opt = popt
print("a = ", a_opt)
print("c = ", c_opt)

perr = np.sqrt(np.diag(pcov))
Da, Dc = perr
print("a = %6.2f +/- %4.2f" % (a_opt, Da))
print("c = %6.2f +/- %4.2f" % (c_opt, Dc))

R2 = np.sum((f_model(xvals, a_opt, c_opt) - yvals.mean())**2) / np.sum((yvals - yvals.mean())**2)
print("r^2 = %10.6f" % R2)

estimates_opt = f_model(x_testarray, a_opt, c_opt)

plt.plot(x_testarray, estimates_opt, label = 'model', marker = '.', linewidth = 0)
plt.plot(xvals, yvals, label = 'data', marker = 'x', linewidth = 0)
plt.legend()


channel_nums_array_nonlinear =  Am_v1_channel_no[0:250]
channel_nums_array_linear = Am_v1_channel_no[251:]

energy_vals_array_nonlinear = f_model(channel_nums_array_nonlinear, a_opt, c_opt)

#%%

# linear section 251 onwards




#%%

np.savetxt('calibrated_energies.csv', energy_vals_array)














