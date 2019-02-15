import numpy as np
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
import json
from pprint import pprint
from scipy.signal import find_peaks


with open('dis_info_04.txt') as f:
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

