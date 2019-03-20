'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

data = np.load('tmp_feat/FFT_bin_fold1.npz')
FFT = data['arr_2']  # shape (7893,80)
time = np.arange(0, FFT.shape[0],1)
freq = np.arange(0, FFT.shape[1],1)
print 'SHAPE: ', FFT.shape
#print FFT.index(max(FFT[0]))

#FFT_trans = np.transpose(FFT)

#Plotting
#print FFT.imag
im = plt.imshow(FFT.real,cmap=cm.get_cmap("rainbow")) # drawing the function
plt.colorbar(im)  

plt.ylim(100, 200)
#plt.xlim(0, 40)
plt.xlabel('freq')
plt.ylabel('time')
plt.title('FFT')


plt.show()


