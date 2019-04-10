from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

data = np.load('feat/mbe_bin_fold4.npz')
mbe = data['arr_2']  # shape (7893,80)
time = np.arange(0, mbe.shape[0],1)
#freq = np.arange(0, mbe.shape[1],1)
print 'mbe shape : ', mbe.shape
print 'mbe max : ', mbe.max()
print 'mbe mean : ', mbe.mean()

mbe_trans = np.transpose(mbe)

print mbe.shape
#im = plt.imshow(mbe,cmap=cm.rainbow) # drawing the function
im = plt.imshow(mbe,cmap=cm.get_cmap("rainbow")) # drawing the function
plt.colorbar()  
#plt.clim(-6,6) 
plt.ylim(14500, 14756)
#plt.xlim(0, 40)
plt.xlabel('mel bands (40 x 2)')
plt.xticks(np.arange(0, 81, 20))
plt.ylabel('T')
plt.title('Log mel-band energy')

#fig = plt.figure()
#plt.plot(time,mbe)

plt.show()


