# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as math
import matplotlib.dates as mdates
from scipy.optimize import curve_fit

#%%

Am_241_v3_df = pd.read_csv('Calibration_data_files/Am_241_v3.csv', skiprows = 2,  names = ['time_s', 'Events_N', 'channel_n', 'Energy_keV', 'rate_r_1/S', 'Voltage_V'])


Am_241_v3_df.plot(x = 'channel_n', y = 'Events_N')

AM_241_v3_events = Am_241_v3_df['Events_N']

AM_241_v3_channel_n = Am_241_v3_df['channel_n']

#%%


plt.plot(AM_241_v3_channel_n, AM_241_v3_events)
plt.show()

# events vs channel number
# find the peak - gives channel number for an energy of  59.6 keV
# do this by fitting a gaussian to the peak 
# to allow for inclusion of error evaluation

# Need to crop out the lower peak for this to work

# Try only taking events from 150 onwards?




#%%

#Gaussian fit

x = AM_241_v3_channel_n[120:220]
y = AM_241_v3_events[120:220]


n = len(x)                          
mean = sum(x*y)/n                   
sigma = sum(y*(x-mean)**2)/n        

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma])

plt.plot(x,y,'b+:',label='data')
plt.plot(x,gaus(x,*popt),'ro:',label='fit')
plt.legend()
plt.title('Am_241_v3_fit')
plt.xlabel('Channel_n')
plt.ylabel('Events_N')
plt.show()


