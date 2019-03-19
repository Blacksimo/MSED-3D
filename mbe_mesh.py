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

data = np.load('feat/mbe_bin_fold2.npz')
mbe = data['arr_0']  # shape (7893,80)
time = np.arange(0, mbe.shape[0],1)
freq = np.arange(0, mbe.shape[1],1)

mbe_trans = np.transpose(mbe)


fig = plt.figure()
#ax = fig.gca(projection='2d')

# Make data.

time, freq = np.meshgrid(time, freq)

im = plt.imshow(mbe[200:400][:],cmap=cm.rainbow) # drawing the function
plt.colorbar(im)  

#plt.ylim(0, 200)

plt.xlabel('freq')
plt.ylabel('time')
plt.title('mbe')


plt.show()


