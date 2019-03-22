import numpy as np
import math
import matplotlib.pyplot as plt

# Plot mesh import
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pickle
import scipy.special

def extract_gcc(_FFT):
    N_resolutions = len(_FFT)
    GCC = [None,None,None]
    # 120,240,480 ms
    for res in range(N_resolutions):
        gcc = np.zeros((_FFT[res].shape[0],60))
        delta_count = 0
        #total Bin
        N = _FFT[res].shape[1]/2
        for delta in range(-29, 31):
            # Time varia per ogni train - test -fold
            for time in range(_FFT[res].shape[0]):
                gcc_sum = 0
                for freq in range(N):  # FFT shape = (time,[freqX1,freqX2)
                    fraction_term = (_FFT[res][time][freq] * np.conjugate(_FFT[res][time][freq+N-1]))/(
                        abs(_FFT[res][time][freq]) * abs(_FFT[res][time][freq+N-1]))
                    exp_term = np.exp((2j*math.pi*freq*delta)/N) 
                    #exp_term = np.real(np.exp(np.complex((2*math.pi*freq*delta))/40))
                    gcc_sum += fraction_term*exp_term
                gcc[time][delta_count] = gcc_sum
            delta_count += 1
            print (delta_count)
        GCC[res]=gcc
        print 'Resolution: ', res, 'gcc shape: ', gcc.shape


    # dovrà essere T x 60 x 3*binom(C,2) --> se due canali --> T x 60 x 3
    # scipy.special.binom(2, 2) = 1 --> 1*3 = 3 ambi
    # scipy.special.binom(4, 2) = 6 --> 6*3 = 18 ambi
    print('shape GCC:', GCC[0].shape, GCC[1].shape, GCC[2].shape)
    return GCC


gcc = extract_gcc('tmp_feat/b093.wav_bin_FFT.npz')
#pickle.dump( gcc, open( "gcc_0_fold1.p", "wb" ) )
fig = plt.figure()
mesh = plt.imshow(gcc, cmap=cm.get_cmap("rainbow"))  # drawing the function
plt.colorbar(mesh)
plt.clim(gcc.min(), gcc.max())

#plt.ylim(0, 200)

plt.xlabel('tau')
plt.ylabel('time')
plt.title('gcc')
plt.show()
