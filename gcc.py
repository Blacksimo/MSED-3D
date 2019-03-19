import numpy as np
import math
import matplotlib.pyplot as plt

def extract_gcc(file):
    data = np.load(file)
    mbe = data['arr_0']  # shape (7893,80)
    print mbe.shape
    #arr_1 = data['arr_1'] #one hot
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
                print gcc_sum
            gcc[time][delta_count] = gcc_sum
        delta_count+=1
        #print delta_count

    print(gcc[1][:])
    return gcc


extract_gcc('feat/a001.wav_bin.npz')


fig = plt.figure()

