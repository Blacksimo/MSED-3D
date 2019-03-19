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

data = np.load('feat/b099.wav_bin.npz')
mbe = data['arr_0']  # shape (7893,80)
time = np.arange(0, mbe.shape[0],1)
freq = np.arange(0, mbe.shape[1],1)

mbe_trans = np.transpose(mbe)


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.

time, freq = np.meshgrid(time, freq)

# Plot the surface.
surf = ax.plot_surface(time, freq, mbe_trans, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
