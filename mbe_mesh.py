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

data = np.load('feat/b093.wav_bin.npz')
mbe = data['arr_0']  # shape (7893,80)
time = np.arange(0, mbe.shape[0],1)
freq = np.arange(0, mbe.shape[1],1)

mbe_trans = np.transpose(mbe)


#fig = plt.figure()
#ax = fig.gca(projection='2d')

# Make data.

#time, freq = np.meshgrid(time, freq)
print mbe.shape
im = plt.imshow(mbe,cmap=cm.rainbow) # drawing the function
plt.colorbar(im)  

#plt.ylim(100, 200)
#plt.xlim(0, 40)
plt.xlabel('freq')
plt.ylabel('time')
plt.title('mbe')


plt.show()


