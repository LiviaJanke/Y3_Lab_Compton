# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as math
import matplotlib.dates as mdates
from scipy.optimize import curve_fit

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

#%%

Am_241_v3_df = pd.read_csv('Calibration_data_files/Am_241_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'Voltage_V'])

Am_241_v4_df = pd.read_csv('Calibration_data_files/Am_241_v4.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'Voltage_V'])

Am_241_v5_df = pd.read_csv('Calibration_data_files/Am_241_v5.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'Voltage_V'])


Am_241_v3_df.plot(x = 'channel_n', y = 'Events_N')
Am_241_v4_df.plot(x = 'channel_n', y = 'Events_N')
Am_241_v5_df.plot(x = 'channel_n', y = 'Events_N')

AM_241_v3_events = Am_241_v3_df['Events_N']
AM_241_v4_events = Am_241_v4_df['Events_N']
AM_241_v5_events = Am_241_v5_df['Events_N']

AM_241_v3_channel_n = Am_241_v3_df['channel_n']
AM_241_v4_channel_n = Am_241_v4_df['channel_n']
AM_241_v5_channel_n = Am_241_v5_df['channel_n']


#%%


plt.plot(AM_241_v3_channel_n, AM_241_v3_events)
plt.plot(AM_241_v4_channel_n, AM_241_v4_events)
plt.plot(AM_241_v5_channel_n, AM_241_v5_events)
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

x = AM_241_v3_channel_n[130:180]
y = AM_241_v3_events[130:180]

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


uncerts_am_241_v3 = (np.sqrt(np.diag(pcov)))

channel_no_am_241_v3 = popt[1]
energy_am_241 = 59.6

channel_no_am_241_v3_uncert = uncerts_am_241_v3[1]

#%%

x = AM_241_v4_channel_n[130:180]
y = AM_241_v4_events[130:180]

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


uncerts_am_241_v4 = (np.sqrt(np.diag(pcov)))

channel_no_am_241_v4 = popt[1]

channel_no_am_241_v4_uncert = uncerts_am_241_v4[1]

#%%

x = AM_241_v5_channel_n[130:180]
y = AM_241_v5_events[130:180]

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


uncerts_am_241_v5 = (np.sqrt(np.diag(pcov)))

channel_no_am_241_v5 = popt[1]

channel_no_am_241_v5_uncert = uncerts_am_241_v5[1]

#%%

channel_nums_am_241 = np.array((channel_no_am_241_v3, channel_no_am_241_v4, channel_no_am_241_v5))

mean_channel_num_am_241 = np.mean(channel_nums_am_241)

channel_nums_am_241_uncerts = np.array((channel_no_am_241_v3_uncert, channel_no_am_241_v4_uncert, channel_no_am_241_v5_uncert))

mean_channel_num_am_241_uncert = np.mean(channel_nums_am_241_uncerts)



#%%

Cs_137_v3_df = pd.read_csv('Calibration_data_files/Cs_137_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'Voltage_V'])


Cs_137_v3_df.plot(x = 'channel_n', y = 'Events_N')

Cs_137_v3_events = Cs_137_v3_df['Events_N']

Cs_137_v3_channel_n = Cs_137_v3_df['channel_n']

plt.plot(Cs_137_v3_channel_n, Cs_137_v3_events)
plt.show()

x = Cs_137_v3_channel_n#[130:180]
y = Cs_137_v3_events#[130:180]

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

channel_no_cs_137_v3 = popt[1]
energy_cs_137_v3 = 662

#%%

Co_57_v3_df = pd.read_csv('Calibration_data_files/Co_57.csv', skiprows = 2567, nrows = 511, names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'Voltage_V'])
#error handling on removing invalid datat


Co_57_v3_df.plot(x = 'channel_n', y = 'Events_N')


Co_57_v3_events = Co_57_v3_df['Events_N']

Co_57_v3_channel_n = Co_57_v3_df['channel_n']

plt.plot(Co_57_v3_channel_n, Co_57_v3_events)
plt.show()

# has somehow recorded all zeros
# very strange
# Am seems to be the only one working so far
# energy of the Cs peak is too high
# Different input voltage needed ? Or problem with the Cassy software?
# Didn't look like all zeroes when measured - maybe an issue with data acquisition?

x = Co_57_v3_channel_n[4:19]
y = Co_57_v3_events[4:19]

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.legend()
plt.title('Co_57 v3 gauss fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.show()


uncerts = (np.sqrt(np.diag(pcov)))

channel_no_Co_57 = popt[1]
energy_Co_57 = 122
energy_Co_57_uncert = uncerts[1]


#%%

energies = np.array((energy_am_241, energy_Co_57))
channels = np.array((mean_channel_num_am_241, channel_no_Co_57))

plt.plot(channels, energies)

# does not look quite right
# unfortunately
# most likely some issue with my code
# try again tomorrow




















