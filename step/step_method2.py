#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
import json
from pprint import pprint
from scipy.signal import find_peaks, peak_prominences


# In[2]:


with open('dis_info/dis_info_04.txt') as f:
    data = json.load(f)

#pprint(data)
tdata = []
vdata = []
keylist = data.keys()
keylist = [int(x) for x in keylist]
keylist.sort()
for key in keylist:
    tdata.append(key)
    vdata.append(data[str(key)]["dis"])
''' 
print(keylist)
plt.plot(tdata, vdata)
plt.xlabel('Frame')
plt.ylabel('Distance')
plt.show()
'''
b, a = signal.butter(3, 0.1, btype='lowpass', analog=False)
low_vdata = signal.filtfilt(b, a, vdata)

plt.plot(tdata, vdata)
plt.plot(tdata, low_vdata, 'r-')
plt.xlabel('Frame')
plt.ylabel('Distance')
plt.show()

v_peaks, _ = find_peaks(vdata, distance=5)
low_peaks, _ = find_peaks(low_vdata, distance=5)

'''
plt.plot(tdata, vdata)
plt.plot(tdata, low_vdata, 'r-')
plt.plot(v_peaks, vdata[v_peaks], "x")
plt.xlabel('Frame')
plt.ylabel('Distance')
plt.show()
'''

print ("v_peaks", v_peaks)
print ("low_peaks", low_peaks)


# In[ ]:





# In[ ]:





# In[3]:


### method2: Calculate mean prominence then use it to compute peaks

# apply for origin data
origin_data = np.array(vdata)
origin_peaks, _ = find_peaks(origin_data)
origin_prominences = peak_prominences(origin_data, origin_peaks)[0]

origin_peak, _ = find_peaks(origin_data, prominence=np.mean(origin_prominences))
print ("origin_peak", origin_peak)

plt.plot(origin_data)
plt.plot(origin_peak, origin_data[origin_peak], "x")
plt.show()


# In[4]:


# apply for lowpassed data
lowpassed_data = np.array(low_vdata)
lowpassed_peaks, _ = find_peaks(lowpassed_data)
lowpassed_prominences = peak_prominences(lowpassed_data, lowpassed_peaks)[0]

lowpassed_peak, _ = find_peaks(lowpassed_data, prominence=np.mean(lowpassed_prominences))
print ("lowpassed_peak", lowpassed_peak)

plt.plot(lowpassed_data)
plt.plot(lowpassed_peak, lowpassed_data[lowpassed_peak], "x")
plt.show()


# In[ ]:





# In[ ]:




