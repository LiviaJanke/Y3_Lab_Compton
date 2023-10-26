# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:10:01 2023

@author: lme19
"""

from __future__ import print_function, division
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

channel_nos = Am_241_625V_v1_df['channel_n']

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

# Taking means of each dataset

Am_channel_nos = (Am_241_625V_v1_df['channel_n'] + Am_241_625V_v2_df['channel_n'] + Am_241_625V_v3_df['channel_n']) / 3
Am_events = (Am_241_625V_v1_df['Events_N'] + Am_241_625V_v2_df['Events_N'] + Am_241_625V_v3_df['Events_N']) / 3

Cs_channel_nos = (Cs_137_625V_df['channel_n'] + Cs_137_625V_v2_df['channel_n'] + Cs_137_625V_v3_df['channel_n']) / 3
Cs_events = (Cs_137_625V_df['Events_N'] + Cs_137_625V_v2_df['Events_N'] + Cs_137_625V_v3_df['Events_N']) / 3

Co_channel_nos = (Co_57_625V_v1_df['channel_n'] + Co_57_625V_v2_df['channel_n'] + Co_57_625V_v3_df['channel_n']) / 3
Co_events = (Co_57_625V_v1_df['Events_N'] + Co_57_625V_v2_df['Events_N'] + Co_57_625V_v3_df['Events_N']) / 3

Ba_channel_nos = (Ba_133_625V_v1_df['channel_n'] + Ba_133_625V_v2_df['channel_n'] + Ba_133_625V_v3_df['channel_n']) / 3
Ba_events = (Ba_133_625V_v1_df['Events_N'] +  Ba_133_625V_v2_df['Events_N'] + Ba_133_625V_v3_df['Events_N']) / 3

Na_channel_nos = (Na_22_625V_v1_df['channel_n'] + Na_22_625V_v2_df['channel_n'] + Na_22_625V_v3_df['channel_n']) / 3
Na_events = (Na_22_625V_v1_df['Events_N'] + Na_22_625V_v2_df['Events_N'] + Na_22_625V_v3_df['Events_N']) / 3


#%%

#Applying peak fit to each mean

Am_channel_no, Am_channel_no_uncert, Am_sigma = Gaussian_peak_fit(Am_channel_nos[40:60], Am_events[40:60])

Cs_channel_no, Cs_channel_no_uncert, Cs_sigma = Gaussian_peak_fit(Cs_channel_nos[385:450], Cs_events[385:450])

Co_channel_no, Co_channel_no_uncert, Co_sigma = Gaussian_peak_fit(Co_channel_nos[84:107], Co_events[84:107])

Ba_channel_no_p1, Ba_channel_no_uncert_p1, Ba_sigma_p1 = Gaussian_peak_fit(Ba_channel_nos[20:40], Ba_events[20:40])

Ba_channel_no_p2, Ba_channel_no_uncert_p2, Ba_sigma_p2 = Gaussian_peak_fit(Ba_channel_nos[50:80], Ba_events[50:80])

Ba_channel_no_p3, Ba_channel_no_uncert_p3, Ba_sigma_p3 = Gaussian_peak_fit(Ba_channel_nos[220:260], Ba_events[220:260])

Na_channel_no, Na_channel_no_uncert, Na_sigma = Gaussian_peak_fit(Na_channel_nos[290:350], Na_events[290:350])


#%%


mean_sigma = (Am_sigma + Cs_sigma + Co_sigma + Ba_sigma_p1 + Ba_sigma_p2 + Ba_sigma_p3 + Na_sigma) / 7

print(mean_sigma)

#%%

energy_Am_241 = 59.54
energy_Co_57 = 122
energy_Cs_137 = 662
energy_Ba_133_p1 = 30.85 
energy_Ba_133_p2 = 81.0
energy_Ba_133_p3 = 356.0
energy_Na_22 = 511


#%%

energies = np.array((energy_Am_241, energy_Co_57, energy_Cs_137, energy_Ba_133_p1, energy_Ba_133_p2, energy_Ba_133_p3, energy_Na_22))

channel_nums = np.array((Am_channel_no, Co_channel_no, Cs_channel_no, Ba_channel_no_p1, Ba_channel_no_p2, Ba_channel_no_p3, Na_channel_no))

channel_num_uncerts = np.array((Am_channel_no_uncert, Co_channel_no_uncert, Cs_channel_no_uncert, Ba_channel_no_uncert_p1, Ba_channel_no_uncert_p2, Ba_channel_no_uncert_p3, Na_channel_no_uncert))

print(energies)
print(channel_nums)
print(channel_num_uncerts)

plt.errorbar(channel_nums, energies, xerr = channel_num_uncerts, linewidth = 0, capsize = 1, marker = 'x', elinewidth = 1, label = 'vals')
plt.legend()

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

# quadratics don@t have poinbts of infelction



#%%

# Splitting calib fit into linear and nonlinear

# Non-linear section 0 to 100


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


channel_nums_array_nonlinear =  channel_nos[0:120]
channel_nums_array_linear = channel_nos[120:]

energy_vals_array_nonlinear = np.array(f_model(channel_nums_array_nonlinear, a_opt, c_opt))

#%%

# linear section 251 onwards

# linear regression fit

X = xvals.reshape(-1, 1)
Y = yvals.reshape(-1, 1)

reg = LinearRegression().fit(X, Y, sample_weight=(xvals_err))


score = reg.score(X, Y, sample_weight=(xvals_err))

coeffs = reg.coef_

intercept = reg.intercept_ * -1


x_testarray = (np.linspace(120, 512, 10000)).reshape(-1, 1)

y_predicted = reg.predict(x_testarray)

plt.plot(x_testarray, y_predicted, label = 'Linear Regression')
plt.errorbar(xvals, yvals, xerr = xvals_err, label = 'data points', capsize = 1, linewidth = 0.1, marker = '.')


plt.ylabel('Energy (keV)')
plt.xlabel('Channel Number')
plt.legend()
plt.title('Linear Regression')
plt.grid()
plt.show()


print(coeffs)
print(intercept)
print(score)


channel_vals_array_linear = (np.linspace(120, 511, 391)).reshape(-1, 1)

energy_vals_array_linear = np.hstack(reg.predict(channel_vals_array_linear))


#%%

total_energies_array = np.hstack((energy_vals_array_nonlinear, energy_vals_array_linear))

#%%

plt.plot(channel_nos, total_energies_array)
plt.errorbar(xvals, yvals, xerr = xvals_err, label = 'data points', capsize = 1, linewidth = 0.1, marker = '.')

#%%

np.savetxt('calibrated_energies_non_and_lin.csv', total_energies_array)


#%%

# making an energies to channels file 
# Maybe later
# Would be a lot more thorough

# Make a Gaussian to convolve with the measured spectra
# using mean sigma and the pyastronomy broadening package

# Set up an input spectrum
x = np.linspace(1, 511, 1000)

# Make a Gaussian

#y  = 















































