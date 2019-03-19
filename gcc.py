import numpy as np
import math
import matplotlib.pyplot as plt

#Plot mesh import
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def extract_gcc(file):
    data = np.load(file)
    mbe = data['arr_0']  # shape (time,frequency), arr_0 or arr_2
    #arr_1-3 = data['arr_1'] #one hot
    print('shape mbe: ', mbe.shape)
    gcc = np.zeros((mbe.shape[0],60))
    delta_count = 0
    for delta in range(-29,31):
        for time in range(mbe.shape[0]):
            gcc_sum = 0
            for freq in range(40):
                exp_term = np.real(np.exp((2j*math.pi*freq*delta)/40))
                #print exp_term
                fraction_term = (mbe[time][freq] * mbe[time][freq+39])/(abs(mbe[time][freq]) * abs(mbe[time][freq+39]))
                gcc_sum += fraction_term*exp_term
            gcc[time][delta_count] = gcc_sum
        delta_count+=1
        print delta_count

    print(gcc.shape)
    return gcc

folders = ['mbe_bin_fold1.npz','mbe_bin_fold2.npz','mbe_bin_fold3.npz','mbe_bin_fold4.npz']
extract_gcc('feat/mbe_bin_fold4.npz')

"""
fig = plt.figure()
mesh = plt.imshow(mbe[0:200][:],cmap=cm.rainbow) # drawing the function
plt.colorbar(mesh)  

#plt.ylim(0, 200)

plt.xlabel('freq')
plt.ylabel('time')
plt.title('mbe')
plt.show()
"""
