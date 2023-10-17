# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:46:55 2023

@author: baidi
"""

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


# first part with least squares
from scipy.optimize import curve_fit

# second part about ODR
from scipy.odr import ODR, Model, Data, RealData

# style and notebook integration of the plots
import seaborn as sns



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

#BA133 
Ba_133_625V_v1_df = pd.read_csv('Calibration_data_files/Ba__133_625V_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])
Ba_133_625V_v2_df = pd.read_csv('Calibration_data_files/Ba__133_625V_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])
Ba_133_625V_v3_df = pd.read_csv('Calibration_data_files/Ba__133_625V_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])

#%%
Ba_133_625V_v1_df.plot(x = 'channel_n', y = 'Events_N')
Ba_133_625V_v2_df.plot(x = 'channel_n', y = 'Events_N')
Ba_133_625V_v3_df.plot(x = 'channel_n', y = 'Events_N')


#%%
#peak 30.85 v1
Ba_133_625V_v1_events = Ba_133_625V_v1_df['Events_N']

Ba_133_625V_v1_channel_n = Ba_133_625V_v1_df['channel_n']

plt.plot(Ba_133_625V_v1_channel_n, Ba_133_625V_v1_events)
plt.show()

x = Ba_133_625V_v1_channel_n[:50]
y = Ba_133_625V_v1_events[:50]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Ba 133 peak 1 v1 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts = (np.sqrt(np.diag(pcov)))

channel_no_Ba_133_625V_peak1_v1 = popt[1]
energy_Ba_133_625V_peak1 = 30.85

channel_no_Ba_133_625V_peak1_v1_uncert = uncerts[1]

#%%
#peak 30.85 v2
Ba_133_625V_v2_events = Ba_133_625V_v2_df['Events_N']

Ba_133_625V_v2_channel_n = Ba_133_625V_v2_df['channel_n']

plt.plot(Ba_133_625V_v2_channel_n, Ba_133_625V_v2_events)
plt.show()

x = Ba_133_625V_v2_channel_n[:50]
y = Ba_133_625V_v2_events[:50]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))


popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Ba 133 peak 1 v2 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts = (np.sqrt(np.diag(pcov)))

channel_no_Ba_133_625V_peak1_v2 = popt[1]
channel_no_Ba_133_625V_peak1_v2_uncert = uncerts[1]
#%%
#peak 30.85 v3
Ba_133_625V_v3_events = Ba_133_625V_v3_df['Events_N']

Ba_133_625V_v3_channel_n = Ba_133_625V_v3_df['channel_n']

plt.plot(Ba_133_625V_v3_channel_n, Ba_133_625V_v3_events)
plt.show()

x = Ba_133_625V_v3_channel_n[:50]
y = Ba_133_625V_v3_events[:50]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))


popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Ba 133 peak 1 v3 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts = (np.sqrt(np.diag(pcov)))

channel_no_Ba_133_625V_peak1_v3 = popt[1]
channel_no_Ba_133_625V_peak1_v3_uncert = uncerts[1]

#%%
#peak 1 mean
channel_nums_Ba_133_625V_peak1 = np.array((channel_no_Ba_133_625V_peak1_v1, channel_no_Ba_133_625V_peak1_v2 , channel_no_Ba_133_625V_peak1_v3))

mean_channel_num_Ba_133_625V_peak1 = np.mean(channel_nums_Ba_133_625V_peak1)

channel_nums_Ba_133_625V_peak1_uncerts = np.array((channel_no_Ba_133_625V_peak1_v1_uncert, channel_no_Ba_133_625V_peak1_v2_uncert, channel_no_Ba_133_625V_peak1_v3_uncert))

mean_channel_num_Ba_133_625V_peak1_uncert = np.mean(channel_nums_Ba_133_625V_peak1_uncerts)

print('Channel Num')
print(mean_channel_num_Ba_133_625V_peak1)
print('Uncert')
print(mean_channel_num_Ba_133_625V_peak1_uncert)
print('Energy')
print(energy_Ba_133_625V_peak1)

#%%
#BA133 Peak 2 81.0
Ba_133_625V_v1_events = Ba_133_625V_v1_df['Events_N']

Ba_133_625V_v1_channel_n = Ba_133_625V_v1_df['channel_n']

plt.plot(Ba_133_625V_v1_channel_n, Ba_133_625V_v1_events)
plt.show()

x = Ba_133_625V_v1_channel_n[50:100]
y = Ba_133_625V_v1_events[50:100]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))


popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Ba 133 peak 2 v1 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts = (np.sqrt(np.diag(pcov)))

channel_no_Ba_133_625V_peak2_v1 = popt[1]
energy_Ba_133_625V_peak2 = 81.0

channel_no_Ba_133_625V_peak2_v1_uncert = uncerts[1]

#%%
#peak 81.0 v2
Ba_133_625V_v2_events = Ba_133_625V_v2_df['Events_N']

Ba_133_625V_v2_channel_n = Ba_133_625V_v2_df['channel_n']

plt.plot(Ba_133_625V_v2_channel_n, Ba_133_625V_v2_events)
plt.show()

x = Ba_133_625V_v2_channel_n[50:100]
y = Ba_133_625V_v2_events[50:100]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Ba 133 peak 2 v2 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts = (np.sqrt(np.diag(pcov)))

channel_no_Ba_133_625V_peak2_v2 = popt[1]
channel_no_Ba_133_625V_peak2_v2_uncert = uncerts[1]
#%%
#peak 81.0 v3
Ba_133_625V_v3_events = Ba_133_625V_v3_df['Events_N']

Ba_133_625V_v3_channel_n = Ba_133_625V_v3_df['channel_n']

plt.plot(Ba_133_625V_v3_channel_n, Ba_133_625V_v3_events)
plt.show()

x = Ba_133_625V_v3_channel_n[50:100]
y = Ba_133_625V_v3_events[50:100]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))


popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Ba 133 peak2 v3 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts = (np.sqrt(np.diag(pcov)))

channel_no_Ba_133_625V_peak2_v3 = popt[1]
channel_no_Ba_133_625V_peak2_v3_uncert = uncerts[1]

#%%
#peak 2 mean
channel_nums_Ba_133_625V_peak2 = np.array((channel_no_Ba_133_625V_peak2_v1, channel_no_Ba_133_625V_peak2_v2 , channel_no_Ba_133_625V_peak2_v3))

mean_channel_num_Ba_133_625V_peak2 = np.mean(channel_nums_Ba_133_625V_peak2)

channel_nums_Ba_133_625V_peak2_uncerts = np.array((channel_no_Ba_133_625V_peak2_v1_uncert, channel_no_Ba_133_625V_peak2_v2_uncert, channel_no_Ba_133_625V_peak2_v3_uncert))

mean_channel_num_Ba_133_625V_peak2_uncert = np.mean(channel_nums_Ba_133_625V_peak2_uncerts)

print('Channel Num')
print(mean_channel_num_Ba_133_625V_peak2)
print('Uncert')
print(mean_channel_num_Ba_133_625V_peak2_uncert)
print('Energy')
print(energy_Ba_133_625V_peak2)
#%%
#BA 133 Peak 3 356.0

Ba_133_625V_v1_events = Ba_133_625V_v1_df['Events_N']

Ba_133_625V_v1_channel_n = Ba_133_625V_v1_df['channel_n']

plt.plot(Ba_133_625V_v1_channel_n, Ba_133_625V_v1_events)
plt.show()

x = Ba_133_625V_v1_channel_n[220:300]
y = Ba_133_625V_v1_events[220:300]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Ba 133 peak 3 v1 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts = (np.sqrt(np.diag(pcov)))

channel_no_Ba_133_625V_peak3_v1 = popt[1]
energy_Ba_133_625V_peak3 = 356.0

channel_no_Ba_133_625V_peak3_v1_uncert = uncerts[1]

#%%
#peak 81.0 v2
Ba_133_625V_v2_events = Ba_133_625V_v2_df['Events_N']

Ba_133_625V_v2_channel_n = Ba_133_625V_v2_df['channel_n']

plt.plot(Ba_133_625V_v2_channel_n, Ba_133_625V_v2_events)
plt.show()

x = Ba_133_625V_v2_channel_n[220:300]
y = Ba_133_625V_v2_events[220:300]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Ba 133 peak 3 v2 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts = (np.sqrt(np.diag(pcov)))

channel_no_Ba_133_625V_peak3_v2 = popt[1]
channel_no_Ba_133_625V_peak3_v2_uncert = uncerts[1]
#%%
#peak 81.0 v3
Ba_133_625V_v3_events = Ba_133_625V_v3_df['Events_N']

Ba_133_625V_v3_channel_n = Ba_133_625V_v3_df['channel_n']

plt.plot(Ba_133_625V_v3_channel_n, Ba_133_625V_v3_events)
plt.show()

x = Ba_133_625V_v3_channel_n[220:300]
y = Ba_133_625V_v3_events[220:300]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Ba 133 peak3 v3 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts = (np.sqrt(np.diag(pcov)))

channel_no_Ba_133_625V_peak3_v3 = popt[1]
channel_no_Ba_133_625V_peak3_v3_uncert = uncerts[1]

#%%
#peak 3 mean
channel_nums_Ba_133_625V_peak3 = np.array((channel_no_Ba_133_625V_peak3_v1, channel_no_Ba_133_625V_peak3_v2 , channel_no_Ba_133_625V_peak3_v3))

mean_channel_num_Ba_133_625V_peak3 = np.mean(channel_nums_Ba_133_625V_peak3)

channel_nums_Ba_133_625V_peak3_uncerts = np.array((channel_no_Ba_133_625V_peak3_v1_uncert, channel_no_Ba_133_625V_peak3_v2_uncert, channel_no_Ba_133_625V_peak3_v3_uncert))

mean_channel_num_Ba_133_625V_peak3_uncert = np.mean(channel_nums_Ba_133_625V_peak3_uncerts)

print('Channel Num')
print(mean_channel_num_Ba_133_625V_peak3)
print('Uncert')
print(mean_channel_num_Ba_133_625V_peak3_uncert)
print('Energy')
print(energy_Ba_133_625V_peak3)

#%%
#Na 22

Na_22_625V_v1_df = pd.read_csv('Calibration_data_files/Na__22_625V_v1.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])
Na_22_625V_v2_df = pd.read_csv('Calibration_data_files/Na__22_625V_v2.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])
Na_22_625V_v3_df = pd.read_csv('Calibration_data_files/Na__22_625V_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'dead_time','Voltage_V'])

#%%
Na_22_625V_v1_df.plot(x = 'channel_n', y = 'Events_N')
Na_22_625V_v2_df.plot(x = 'channel_n', y = 'Events_N')
Na_22_625V_v3_df.plot(x = 'channel_n', y = 'Events_N')

#%%

Na_22_625V_v1_events = Na_22_625V_v1_df['Events_N']

Na_22_625V_v1_channel_n = Na_22_625V_v1_df['channel_n']

plt.plot(Na_22_625V_v1_channel_n, Na_22_625V_v1_events)
plt.show()

x = Na_22_625V_v1_channel_n[290:350]
y = Na_22_625V_v1_events[290:350]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Na 22 v1 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts = (np.sqrt(np.diag(pcov)))

channel_no_Na_22_625V_v1 = popt[1]
energy_Na_22_625V = 511

channel_no_Na_22_625V_v1_uncert = uncerts[1]

#%%

Na_22_625V_v2_events = Na_22_625V_v2_df['Events_N']

Na_22_625V_v2_channel_n = Na_22_625V_v2_df['channel_n']

plt.plot(Na_22_625V_v2_channel_n, Na_22_625V_v2_events)
plt.show()

x = Na_22_625V_v2_channel_n[290:350]
y = Na_22_625V_v2_events[290:350]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))


popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Na 22 v2 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts = (np.sqrt(np.diag(pcov)))

channel_no_Na_22_625V_v2 = popt[1]

channel_no_Na_22_625V_v2_uncert = uncerts[1]

#%%

Na_22_625V_v3_events = Na_22_625V_v3_df['Events_N']

Na_22_625V_v3_channel_n = Na_22_625V_v3_df['channel_n']

plt.plot(Na_22_625V_v3_channel_n, Na_22_625V_v3_events)
plt.show()

x = Na_22_625V_v3_channel_n[290:350]
y = Na_22_625V_v3_events[290:350]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))


popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Na 22 v3 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts = (np.sqrt(np.diag(pcov)))

channel_no_Na_22_625V_v3 = popt[1]

channel_no_Na_22_625V_v3_uncert = uncerts[1]



#%%

channel_nums_Na_22_625V = np.array((channel_no_Na_22_625V_v1, channel_no_Na_22_625V_v2 , channel_no_Na_22_625V_v3))

mean_channel_num_Na_22_625V = np.mean(channel_nums_Na_22_625V)

channel_nums_Na_22_625V_uncerts = np.array((channel_no_Na_22_625V_v1_uncert, channel_no_Na_22_625V_v2_uncert, channel_no_Na_22_625V_v3_uncert))

mean_channel_num_Na_22_625V_uncert = np.mean(channel_nums_Na_22_625V_uncerts)

print('Channel Num')
print(mean_channel_num_Na_22_625V)
print('Uncert')
print(mean_channel_num_Na_22_625V_uncert)
print('Energy')
print(energy_Na_22_625V)

#%%



energies = np.array((energy_Am_241_625V_v1, energy_Co_57_625V, energy_Cs_137_625V, energy_Ba_133_625V_peak1, energy_Ba_133_625V_peak2, energy_Ba_133_625V_peak3, energy_Na_22_625V))

channel_nums = np.array((mean_channel_num_am_241, mean_channel_num_Co_57_625V, mean_channel_num_Cs_137_625V, mean_channel_num_Ba_133_625V_peak1, mean_channel_num_Ba_133_625V_peak2, mean_channel_num_Ba_133_625V_peak3, mean_channel_num_Na_22_625V))

channel_num_uncerts = np.array((mean_channel_num_am_241_uncert, mean_channel_num_Co_57_625V_uncert, mean_channel_num_Cs_137_625V_uncert, mean_channel_num_Ba_133_625V_peak1_uncert, mean_channel_num_Ba_133_625V_peak2_uncert, mean_channel_num_Ba_133_625V_peak3_uncert, mean_channel_num_Na_22_625V_uncert))

print(energies)
print(channel_nums)
print(channel_num_uncerts)

#%%


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


plt.ylabel('Energy (keV)')
plt.xlabel('Channel Number')
plt.legend()
plt.title('Linear Regression')
plt.grid()
plt.show()


print(coeffs)
print(intercept)
print(score)

#%%

# save the output data points as a csv file
df = pd.read_csv("channel_nums_energies.csv", skiprows = 1, names = ['x', 'y', 'Dx'])


#%%

# curve fit to include this non-linearity in the energy calibration

# the curve is only at low energies - fit to this section?


#def f_model(x, a, c):
#    return np.log((a + x)**2 / (x - c)**2)

# making a new model function that fits to the data

def f_model(x, a, c):
    return (a * x) + (c * x**2)

estimate = f_model(x_testarray, 1.6, 1e-9)

plt.plot(x_testarray, estimate, label = 'model', marker = '.', linewidth = 0)
plt.plot(xvals, yvals, label = 'data', marker = 'x', linewidth = 0)
plt.legend()



#%%

# running curve fit routine


popt, pcov = curve_fit(
    f=f_model,       # model function
    xdata=xvals,   # x data
    ydata=yvals,   # y data
    p0=(1.6, 1e-9),      # initial value of the parameters
    sigma= xvals_err   # uncertainties on y
)

print(popt)


#%%

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

#%%

# applying the calibration to the channel numbers

channel_nums_array =  Am_241_625V_v1_channel_n

energy_vals_array = f_model(channel_nums_array, a_opt, c_opt)

np.savetxt('calibrated_energies.csv', energy_vals_array)

#%%






















