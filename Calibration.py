# -*- coding: utf-8 -*-


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

#%%

Am_241_625V_v1_df = pd.read_csv('Calibration_data_files/Am__241_625V_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time', 'Voltage_V'])

Am_241_625V_v2_df = pd.read_csv('Calibration_data_files/Am__241_625V_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])

Am_241_625V_v3_df = pd.read_csv('Calibration_data_files/Am__241_625V_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])


Am_241_625V_v1_df.plot(x = 'channel_n', y = 'Events_N')
Am_241_625V_v2_df.plot(x = 'channel_n', y = 'Events_N')
Am_241_625V_v3_df.plot(x = 'channel_n', y = 'Events_N')

Am_241_625V_v1_events = Am_241_625V_v1_df['Events_N']
Am_241_625V_v2_events = Am_241_625V_v2_df['Events_N']
Am_241_625V_v3_events = Am_241_625V_v3_df['Events_N']

Am_241_625V_v1_channel_n = Am_241_625V_v1_df['channel_n']
Am_241_625V_v2_channel_n = Am_241_625V_v2_df['channel_n']
Am_241_625V_v3_channel_n = Am_241_625V_v3_df['channel_n']


#%%


plt.plot(Am_241_625V_v1_channel_n, Am_241_625V_v1_events)
plt.plot(Am_241_625V_v2_channel_n, Am_241_625V_v2_events)
plt.plot(Am_241_625V_v3_channel_n, Am_241_625V_v3_events)
plt.show()

# events vs channel number
# find the peak - gives channel number for an energy of  59.6 keV
# do this by fitting a gaussian to the peak 
# to allow for inclusion of error evaluation

# Need to crop out the lower peak for this to work

# Try only taking events from 150 onwards?




#%%

#Gaussian fit

# for each data set for Am

x = Am_241_625V_v1_channel_n[40:60]
y =Am_241_625V_v1_events[40:60]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))



popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Am 241 v3 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts_Am_241_625V_v1 = (np.sqrt(np.diag(pcov)))

channel_no_Am_241_625V_v1 = popt[1]
energy_Am_241_625V_v1 = 59.54

channel_no_Am_241_625V_v1_uncert = uncerts_Am_241_625V_v1[1]

#%%

x = Am_241_625V_v2_channel_n[40:60]
y = Am_241_625V_v2_events[40:60]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))



popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Am 241 v1 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts_Am_241_625V_v2 = (np.sqrt(np.diag(pcov)))

channel_no_Am_241_625V_v2 = popt[1]

channel_no_Am_241_625V_v2_uncert = uncerts_Am_241_625V_v2[1]

#%%

x = Am_241_625V_v3_channel_n[40:60]
y = Am_241_625V_v3_events[40:60]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))



popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Am 241 v2 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts_Am_241_625V_v3 = (np.sqrt(np.diag(pcov)))

channel_no_Am_241_625V_v3 = popt[1]

channel_no_Am_241_625V_v3_uncert = uncerts_Am_241_625V_v3[1]

#%%

channel_nums_am_241 = np.array((channel_no_Am_241_625V_v1, channel_no_Am_241_625V_v2, channel_no_Am_241_625V_v3))

mean_channel_num_am_241 = np.mean(channel_nums_am_241)

channel_nums_am_241_uncerts = np.array((channel_no_Am_241_625V_v1_uncert, channel_no_Am_241_625V_v2_uncert, channel_no_Am_241_625V_v3_uncert))

mean_channel_num_am_241_uncert = np.mean(channel_nums_am_241_uncerts)

print('Channel Num')
print(mean_channel_num_am_241)
print('Uncert')
print(mean_channel_num_am_241_uncert)
print('Energy')
print(energy_Am_241_625V_v1)

#%%

Cs_137_625V_df = pd.read_csv('Calibration_data_files/Cs_137_625V.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])
Cs_137_625V_v2_df = pd.read_csv('Calibration_data_files/Cs_137_625V_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])
Cs_137_625V_v3_df = pd.read_csv('Calibration_data_files/Cs_137_625V_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])

#%%
Cs_137_625V_df.plot(x = 'channel_n', y = 'Events_N')
Cs_137_625V_v2_df.plot(x = 'channel_n', y = 'Events_N')
Cs_137_625V_v3_df.plot(x = 'channel_n', y = 'Events_N')

#%%

Cs_137_625V_events = Cs_137_625V_df['Events_N']

Cs_137_625V_channel_n = Cs_137_625V_df['channel_n']

plt.plot(Cs_137_625V_channel_n, Cs_137_625V_events)
plt.show()

x = Cs_137_625V_channel_n[385:450]
y = Cs_137_625V_events[385:450]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Cs_137 v1 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts = (np.sqrt(np.diag(pcov)))

channel_no_Cs_137_625V = popt[1]
energy_Cs_137_625V = 662

channel_no_Cs_137_625V_uncert = uncerts[1]

#%%

Cs_137_625V_v2_events = Cs_137_625V_v2_df['Events_N']

Cs_137_625V_v2_channel_n = Cs_137_625V_v2_df['channel_n']

plt.plot(Cs_137_625V_v2_channel_n, Cs_137_625V_v2_events)
plt.show()

x = Cs_137_625V_v2_channel_n[385:450]
y = Cs_137_625V_v2_events[385:450]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Cs_137 v2 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts = (np.sqrt(np.diag(pcov)))

channel_no_Cs_137_625V_v2 = popt[1]

channel_no_Cs_137_625V_v2_uncert = uncerts[1]

#%%

Cs_137_625V_v3_events = Cs_137_625V_v3_df['Events_N']

Cs_137_625V_v3_channel_n = Cs_137_625V_v3_df['channel_n']

plt.plot(Cs_137_625V_v3_channel_n, Cs_137_625V_v3_events)
plt.show()

x = Cs_137_625V_v3_channel_n[385:450]
y = Cs_137_625V_v3_events[385:450]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Cs_137 v3 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts = (np.sqrt(np.diag(pcov)))

channel_no_Cs_137_625V_v3 = popt[1]

channel_no_Cs_137_625V_v3_uncert = uncerts[1]

#%%

channel_nums_Cs_137_625V = np.array((channel_no_Cs_137_625V, channel_no_Cs_137_625V_v2 , channel_no_Cs_137_625V_v3))

mean_channel_num_Cs_137_625V = np.mean(channel_nums_Cs_137_625V)

channel_nums_Cs_137_625V_uncerts = np.array((channel_no_Cs_137_625V_uncert, channel_no_Cs_137_625V_v2_uncert, channel_no_Cs_137_625V_v3_uncert))

mean_channel_num_Cs_137_625V_uncert = np.mean(channel_nums_Cs_137_625V_uncerts)

print('Channel Num')
print(mean_channel_num_Cs_137_625V)
print('Uncert')
print(mean_channel_num_Cs_137_625V_uncert)
print('Energy')
print(energy_Cs_137_625V)



#%%

#Co 

Co_57_625V_v1_df = pd.read_csv('Calibration_data_files/Co_57_625V_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])
Co_57_625V_v2_df = pd.read_csv('Calibration_data_files/Co_57_625V_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])
Co_57_625V_v3_df = pd.read_csv('Calibration_data_files/Co_57_625V_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])

#%%
Co_57_625V_v1_df.plot(x = 'channel_n', y = 'Events_N')
Co_57_625V_v2_df.plot(x = 'channel_n', y = 'Events_N')
Co_57_625V_v3_df.plot(x = 'channel_n', y = 'Events_N')

#%%

Co_57_625V_v1_events = Co_57_625V_v1_df['Events_N']

Co_57_625V_v1_channel_n = Co_57_625V_v1_df['channel_n']

plt.plot(Co_57_625V_v1_channel_n, Co_57_625V_v1_events)
plt.show()

x = Co_57_625V_v1_channel_n[84:107]
y = Co_57_625V_v1_events[84:107]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Co 57 v1 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts = (np.sqrt(np.diag(pcov)))

channel_no_Co_57_625V_v1 = popt[1]
energy_Co_57_625V = 122

channel_no_Co_57_625V_v1_uncert = uncerts[1]

#%%

Co_57_625V_v2_events = Co_57_625V_v2_df['Events_N']

Co_57_625V_v2_channel_n = Co_57_625V_v2_df['channel_n']

plt.plot(Co_57_625V_v2_channel_n, Co_57_625V_v2_events)
plt.show()

x = Co_57_625V_v2_channel_n[84:107]
y = Co_57_625V_v2_events[84:107]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Co 57 v2 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts = (np.sqrt(np.diag(pcov)))

channel_no_Co_57_625V_v2 = popt[1]

channel_no_Co_57_625V_v2_uncert = uncerts[1]

#%%

Co_57_625V_v3_events = Co_57_625V_v3_df['Events_N']

Co_57_625V_v3_channel_n = Co_57_625V_v3_df['channel_n']

plt.plot(Co_57_625V_v3_channel_n, Co_57_625V_v3_events)
plt.show()

x = Co_57_625V_v3_channel_n[84:107]
y = Co_57_625V_v3_events[84:107]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Co 57 v3 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts = (np.sqrt(np.diag(pcov)))

channel_no_Co_57_625V_v3 = popt[1]

channel_no_Co_57_625V_v3_uncert = uncerts[1]



#%%

channel_nums_Co_57_625V = np.array((channel_no_Co_57_625V_v1, channel_no_Co_57_625V_v2 , channel_no_Co_57_625V_v3))

mean_channel_num_Co_57_625V = np.mean(channel_nums_Co_57_625V)

channel_nums_Co_57_625V_uncerts = np.array((channel_no_Co_57_625V_v1_uncert, channel_no_Co_57_625V_v2_uncert, channel_no_Co_57_625V_v3_uncert))

mean_channel_num_Co_57_625V_uncert = np.mean(channel_nums_Co_57_625V_uncerts)

print('Channel Num')
print(mean_channel_num_Co_57_625V)
print('Uncert')
print(mean_channel_num_Co_57_625V_uncert)
print('Energy')
print(energy_Co_57_625V)




#%%



energies = np.array((energy_Am_241_625V_v1, energy_Co_57_625V, energy_Cs_137_625V))

channel_nums = np.array((mean_channel_num_am_241, mean_channel_num_Co_57_625V, mean_channel_num_Cs_137_625V))

channel_num_uncerts = np.array((mean_channel_num_am_241_uncert, mean_channel_num_Co_57_625V_uncert, mean_channel_num_Cs_137_625V_uncert))

print(energies)
print(channel_nums)
print(channel_num_uncerts)

#%%


plt.plot(energies, channel_nums)


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

# linear regression fit

X = xvals.reshape(-1, 1)
Y = yvals.reshape(-1, 1)

reg = LinearRegression().fit(X, Y, sample_weight=(xvals_err))


score = reg.score(X, Y, sample_weight=(xvals_err))

coeffs = reg.coef_

intercept = reg.intercept_ * -1


x_testarray = (np.linspace(0, 512, 10000)).reshape(-1, 1)

y_predicted = reg.predict(x_testarray)

plt.plot(x_testarray, y_predicted, label = 'Linear Regression')
plt.errorbar(xvals, yvals, xerr = xvals_err, label = 'data points', capsize = 1, linewidth = 0.1, marker = '.')


plt.ylabel('Energyu (keV)')
plt.xlabel('Channel Number')
plt.legend()
plt.title('Linear Regression')
plt.grid()
plt.show()


print(coeffs)
print(intercept)
print(score)














































