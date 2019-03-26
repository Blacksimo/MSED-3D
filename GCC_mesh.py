from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

data = np.load('feat_gcc_th/GCC_480_bin_fold1.npz')
GCC = data['arr_0']  # shape (time,freq)
print 'GCC shape : ', GCC.shape
print 'GCC max : ', GCC.max()
print 'GCC mean : ', GCC.mean()

#GCC_trans = np.transpose(GCC)

#Plotting
#print FFT.imag
im = plt.imshow(GCC,cmap=cm.get_cmap("rainbow")) # drawing the function
plt.colorbar()
plt.clim(-30,30)  

#plt.xlim(0, 40)
#plt.ylim(100, 200)

plt.xlabel('tau')
plt.ylabel('time')
plt.title('GCC')


plt.show()


