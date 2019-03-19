import numpy as np
import math
import matplotlib.pyplot as plt

#Plot mesh import
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pickle
import scipy.special

def extract_gcc(file):
    data = np.load(file)
    mbe = data['arr_0'][0:10][:]  # shape (time,frequency), arr_0 or arr_2
    print('shape mbe: ', mbe.shape)
    gcc = np.zeros((mbe.shape[0],60))
    delta_count = 0
    #Range di 60
    for delta in range(-29,31):
        #Time varia per ogni train - test -fold
        for time in range(mbe.shape[0]):
            gcc_sum = 0
            for freq in range(40):
                fraction_term = (mbe[time][freq] * mbe[time][freq+39])/(abs(mbe[time][freq]) * abs(mbe[time][freq+39]))
                exp_term = np.real(np.exp((2j*math.pi*freq*delta)/40))
                #exp_term = np.real(np.exp(np.complex((2*math.pi*freq*delta))/40))
                gcc_sum += fraction_term*exp_term

            gcc[time][delta_count] = gcc_sum
        delta_count+=1
        print (delta_count)
    # dovrà essere T x 60 x 3*binom(C,2) --> se due canali --> T x 60 x 3
    #scipy.special.binom(2, 2) = 1 --> 1*3 = 3 ambi
    #scipy.special.binom(4, 2) = 6 --> 6*3 = 18 ambi
    print "gcc shape:", gcc.shape 
    return gcc

gcc = extract_gcc('feat/mbe_bin_fold1.npz')       #'mbe_bin_fold1.npz','mbe_bin_fold2.npz','mbe_bin_fold3.npz','mbe_bin_fold4.npz']
#pickle.dump( gcc, open( "gcc_0_fold1.p", "wb" ) )
fig = plt.figure()
mesh = plt.imshow(gcc,cmap=cm.rainbow) # drawing the function
plt.colorbar(mesh)  

#plt.ylim(0, 200)

plt.xlabel('freq')
plt.ylabel('time')
plt.title('gcc')
plt.show()

