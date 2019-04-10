from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

data_120 = np.load('feat_gcc_norm/GCC_120_bin_fold4.npz')
GCC_120 = data_120['arr_2']  # shape (time,freq)

data_240= np.load('feat_gcc_norm/GCC_240_bin_fold4.npz')
GCC_240 = data_240['arr_2']  # shape (time,freq)

data_480 = np.load('feat_gcc_norm/GCC_480_bin_fold4.npz')
GCC_480 = data_480['arr_2']  # shape (time,freq)

print 'GCC max : ', GCC_120.max(), GCC_240.max() ,GCC_480.max() 
print 'GCC mean : ', GCC_120.mean(), GCC_240.mean() ,GCC_480.mean() 

plt.figure(figsize=(15, 10))
plt.subplot(1, 3, 1)
im = plt.imshow(GCC_120,cmap=cm.get_cmap("rainbow")) # drawing the function
plt.colorbar()
#plt.clim(-1.5,1)  
plt.ylim(14500,14756)
plt.xlabel('tau')
plt.xticks(np.arange(0, 61, 20))
plt.ylabel('T')
plt.title('120 ms')
plt.subplot(1, 3, 2)
im = plt.imshow(GCC_240,cmap=cm.get_cmap("rainbow")) # drawing the function
plt.colorbar()
#plt.clim(-1.5,1)  
plt.ylim(14500,14756)
plt.xlabel('tau')
plt.xticks(np.arange(0, 61, 20))
plt.ylabel('T')
plt.title('240 ms')
plt.subplot(1, 3, 3)
im = plt.imshow(GCC_480,cmap=cm.get_cmap("rainbow")) # drawing the function
plt.colorbar()
#plt.clim(-1.5,1)  
plt.ylim(14500,14756)
plt.xlabel('tau')
plt.xticks(np.arange(0, 61, 20))
plt.ylabel('T')
plt.title('480 ms')
plt.show()
        

