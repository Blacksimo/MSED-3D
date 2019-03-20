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
    FFT = data['arr_0'][0:10][:]  # shape (time,frequency), arr_0 or arr_2
    print('shape FFT: ', FFT.shape)
    gcc = np.zeros((FFT.shape[0],60))
    delta_count = 0
    #Range di 60
    N = FFT.shape[1]/2

    for delta in range(-29,31):
        #Time varia per ogni train - test -fold
        for time in range(FFT.shape[0]):
            gcc_sum = 0
            for freq in range(N): #FFT shape = (time,[freqX1,freqX2)
                fraction_term = (FFT[time][freq] * np.conjugate(FFT[time][freq+N-1]))/(abs(FFT[time][freq]) * abs(FFT[time][freq+N-1]))
                exp_term = np.real(np.exp((2j*math.pi*freq*delta)/N))   #TODO N = 1 
                #exp_term = np.real(np.exp(np.complex((2*math.pi*freq*delta))/40))
                gcc_sum += fraction_term*exp_term
            gcc[time][delta_count] = gcc_sum
        delta_count+=1
        print (delta_count)
    # dovrà essere T x 60 x 3*binom(C,2) --> se due canali --> T x 60 x 3
    #scipy.special.binom(2, 2) = 1 --> 1*3 = 3 ambi
    #scipy.special.binom(4, 2) = 6 --> 6*3 = 18 ambi
    print('shape gcc:', gcc.shape)
    return gcc

gcc = extract_gcc('tmp_feat/FFT_bin_fold1.npz')      
#pickle.dump( gcc, open( "gcc_0_fold1.p", "wb" ) )
fig = plt.figure()
mesh = plt.imshow(gcc,cmap=cm.get_cmap("rainbow")) # drawing the function 
plt.colorbar(mesh)  

#plt.ylim(0, 200)

plt.xlabel('tau')
plt.ylabel('time')
plt.title('gcc')
plt.show()

