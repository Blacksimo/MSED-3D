# coding=utf-8
from __future__ import division
import wave
import numpy as np
import utils
import librosa
from IPython import embed
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
import librosa.display
import math
import shutil
import sys
import multiprocessing as mp
from multiprocessing import Process, Queue
import time
from tqdm import tqdm
from p_tqdm import p_map

def load_audio(filename, mono=True, fs=44100):
    

    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        _audio_file = wave.open(filename)

        # Audio info
        sample_rate = _audio_file.getframerate()
        sample_width = _audio_file.getsampwidth()
        number_of_channels = _audio_file.getnchannels()
        number_of_frames = _audio_file.getnframes()

        # Read raw bytes
        data = _audio_file.readframes(number_of_frames)
        _audio_file.close()

        # Convert bytes based on sample_width
        num_samples, remainder = divmod(
            len(data), sample_width * number_of_channels)
        if remainder > 0:
            raise ValueError(
                'The length of data is not a multiple of sample size * number of channels.')
        if sample_width > 4:
            raise ValueError('Sample size cannot be bigger than 4 bytes.')

        if sample_width == 3:
            # 24 bit audio
            a = np.empty((num_samples, number_of_channels, 4), dtype=np.uint8)
            raw_bytes = np.fromstring(data, dtype=np.uint8)
            a[:, :, :sample_width] = raw_bytes.reshape(
                -1, number_of_channels, sample_width)
            a[:, :, sample_width:] = (
                a[:, :, sample_width - 1:sample_width] >> 7) * 255
            audio_data = a.view('<i4').reshape(a.shape[:-1]).T
        else:
            # 8 bit samples are stored as unsigned ints; others as signed ints.
            dt_char = 'u' if sample_width == 1 else 'i'
            a = np.fromstring(data, dtype='<%s%d' % (dt_char, sample_width))
            audio_data = a.reshape(-1, number_of_channels).T

        if mono:
            # Down-mix audio
            audio_data = np.mean(audio_data, axis=0)

        # Convert int values into float
        audio_data = audio_data / float(2 ** (sample_width * 8 - 1) + 1)

        # Resample
        if fs != sample_rate:
            audio_data = librosa.core.resample(audio_data, sample_rate, fs)
            sample_rate = fs
        return audio_data, sample_rate
    return None, None


def load_desc_file(_desc_file):
    _desc_dict = dict()
    for line in open(_desc_file):
        words = line.strip().split('\t')
        name = words[0].split('/')[-1]
        if name not in _desc_dict:
            _desc_dict[name] = list()
        _desc_dict[name].append(
            [float(words[2]), float(words[3]), __class_labels[words[-1]]])
    return _desc_dict


###########################################################
#EXTRACT GCC
###########################################################
def extract_gcc(_FFT,_res, _output,_bar):

    time_cut = _res
    #----
    #TIME
    #---------------------------------------
    time = np.arange(0,60,1)# per le prove
    #time = np.arange(0,_FFT.shape[0],1)
    #--------------------------------------
    #12 CORES, 4 processes for each resolution
    partial_index = time.shape[0]//4


    if time_cut <= 3:
       time = time[0:partial_index]
    elif time_cut <= 6:
        time = time[partial_index:partial_index*2]
    elif time_cut <= 9:
        time = time[partial_index*2:partial_index*3]
    else:
        time = time[partial_index*3:]
        
    TAU = np.arange(-29,31,1)
    gcc = np.zeros((time.shape[0],len(TAU)))
    progress_bar_delta = 0
    #Total bins
    N = _FFT.shape[1]//2
    for delta in TAU: #-29, -28, ..., 28, 29, 30
        #progress_bar_delta +=1
        # Time varia per ogni train - test -fold
        for i,t in enumerate(time):
            gcc_sum = 0
            for freq in range(N): 
                fraction_term = (_FFT[t][freq] * np.conjugate(_FFT[t][freq+N-1]))/(abs(_FFT[t][freq]) * abs(_FFT[t][freq+N-1]))
                exp_term = np.exp((2j*math.pi*freq*delta)/N) 
                #exp_term = np.real(np.exp(np.complex((2*math.pi*freq*delta))/40))
                gcc_sum += fraction_term*exp_term
                
            gcc[i][delta] = gcc_sum
        #print (delta)
        _bar.update(1)
        #progress(progress_bar_delta, 60, status=str(_res))
    _bar.close()
    #print('gcc shape: ', gcc.shape)

    _output.put((_res,gcc))
    
    #return gcc
    

###########################################################
#EXTRACT MBE
###########################################################
def extract_mbe(_y, _sr, _nfft, _nb_mel):
    FFT_120 = None
    FFT_240 = None
    FFT_480 = None
    #--------------------
    # Extract FFT for gcc
    #--------------------
    if not is_mono:
        FFT_120 = librosa.core.stft(
            y=_y, n_fft=_nfft, hop_length=_nfft//2, win_length=8)  # 1/0.12
        FFT_240 = librosa.core.stft(
            y=_y, n_fft=_nfft, hop_length=_nfft//2, win_length=4)
        FFT_480 = librosa.core.stft(
            y=_y, n_fft=_nfft, hop_length=_nfft//2, win_length=2)
        #TRANSPOSE new_shape= (time,freq)
        FFT_120=FFT_120.T
        FFT_240=FFT_240.T
        FFT_480=FFT_480.T
        #FFT =[FFT_120, FFT_240, FFT_480]
        """
        #Plot operations
        D_120 = librosa.amplitude_to_db(np.abs(FFT_120), ref=np.max)
        D_240 = librosa.amplitude_to_db(np.abs(FFT_240), ref=np.max)
        D_480 = librosa.amplitude_to_db(np.abs(FFT_480), ref=np.max)
        plt.figure(figsize=(20, 20))
        plt.subplot(1, 3, 1)
        librosa.display.specshow(D_120, sr=44100)
        plt.colorbar()
        plt.xlabel('time')
        plt.title('120')
        plt.subplot(1, 3, 2)
        librosa.display.specshow(D_240, sr=44100)
        plt.colorbar()
        plt.xlabel('time')
        plt.title('240')
        plt.subplot(1, 3, 3)
        librosa.display.specshow(D_480, sr=44100)
        plt.colorbar()
        plt.xlabel('time')
        plt.title('480')
        plt.show()
        """
    #-------------------------------
    # extract mel band
    #-------------------------------
    spec, n_fft = librosa.core.spectrum._spectrogram(
        y=_y, n_fft=_nfft, hop_length=_nfft//2, power=1)
    # mel_basis è un filtro che si applica all'fft, per ottenere la mel band
    mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel)
    # applicamio il filtro e facciamo il logaritmo
    return np.log(np.dot(mel_basis, spec)), FFT_120, FFT_240, FFT_480


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  




# ###################################################################
#              Main script starts here
# ###################################################################

#RESOLUTIONS = ['120']
RESOLUTIONS = ['120','240','480']
processed_audio_count = 0
is_mono = False
__class_labels = {
    'brakes squeaking': 0,
    'car': 1,
    'children': 2,
    'large vehicle': 3,
    'people speaking': 4,
    'people walking': 5
}

# location of data.
folds_list = [1, 2, 3, 4]
evaluation_setup_folder = '../TUT-sound-events-2017-development/evaluation_setup'
audio_folder = '../TUT-sound-events-2017-development/audio/street'

# Output
feat_folder = 'feat_gcc_th2/'
utils.create_folder(feat_folder)

# User set parameters
nfft = 2048
win_len = nfft 
hop_len = win_len / 2
nb_mel_bands = 40
sr = 44100

# -----------------------------------------------------------------------
# Feature extraction and label generation
# -----------------------------------------------------------------------
# Load labels
train_file = os.path.join(evaluation_setup_folder,
                          'street_fold{}_train.txt'.format(1))
evaluate_file = os.path.join(
    evaluation_setup_folder, 'street_fold{}_evaluate.txt'.format(1))
desc_dict = load_desc_file(train_file)
# contains labels for all the audio in the dataset
desc_dict.update(load_desc_file(evaluate_file))

# Extract features for all audio files, and save it along with labels
for audio_filename in os.listdir(audio_folder):
    audio_file = os.path.join(audio_folder, audio_filename)
    print('Extracting features and label for : {}'.format(audio_file))
    processed_audio_count+=1
    print ('> Processed_audio_count: {}'.format(processed_audio_count) )
    start_time = time.clock()
    print start_time, "time START seconds"
    y, sr = load_audio(audio_file, mono=is_mono, fs=sr)
    mbe = None
    FFT = [None,None,None]

    if is_mono:
        # shape = (freq, time)
        mbe,  FFT_120, FFT_240, FFT_480 = extract_mbe(y, sr, nfft, nb_mel_bands)
        mbe = mbe.T  # shape = (time, freq)
        #FFT NON MI SERVE PERCHE NON CALCOLO GCC
    else:
        # SONO 2 CANALI
        for ch in range(y.shape[0]):
            #print 'CH: ', ch
            mbe_ch, FFT_120_ch, FFT_240_ch, FFT_480_ch = extract_mbe(y[ch, :], sr, nfft, nb_mel_bands)
            mbe_ch = mbe_ch.T
            #FFT è gia trasposto
            if mbe is None:
                mbe = mbe_ch
                FFT_120=FFT_120_ch
                FFT_240=FFT_240_ch
                FFT_480=FFT_480_ch
            else:
                mbe = np.concatenate((mbe, mbe_ch), 1)
                FFT_120 = np.concatenate((FFT_120, FFT_120_ch), 1)
                FFT_240 = np.concatenate((FFT_240, FFT_240_ch), 1)
                FFT_480 = np.concatenate((FFT_480, FFT_480_ch), 1)
        print('> FFT extracted for both channels')
    if not is_mono:
        multiprocess_diz = {1 :FFT_120, 2 :FFT_240, 3 :FFT_480,   #1/4
                            4 :FFT_120, 5 :FFT_240, 6 :FFT_480,   #2/4
                            7 :FFT_120, 8 :FFT_240, 9 :FFT_480,   #3/4
                            10 :FFT_120, 11 :FFT_240, 12 :FFT_480}  #4/4
        
        output = mp.Queue()
        
        processes = [mp.Process(target=extract_gcc, args=(multiprocess_diz[res],res,output,tqdm(total=60,position=res)))for res in range(1,13)]
        
        # Run processes
        for p in processes:
            p.start()
        print('Processes started')

        results =  [output.get() for p in processes]

        # Exit the completed processes
        #print('Processes joined')
        for p in processes:
            p.join()
            #print("Join on process ", p)
     
        # Get process results from the output queue
        #results =  [output.get() for p in processes]
        results.sort()
        #to check the order 
        for results_res_order in results:
            print(results_res_order[0])
        results = [r[1] for r in results]
         
        GCC_120 = np.concatenate((results[0], results[3],results[6],results[9]), 0)   
        GCC_240 = np.concatenate((results[1], results[4],results[7],results[10]), 0)  
        GCC_480 = np.concatenate((results[2], results[5],results[8],results[11]), 0)  
    print time.clock() - start_time, "seconds"
 
    print "GCC_120: ", GCC_120.shape
    print "GCC_240: ", GCC_240.shape
    print "GCC_480: ", GCC_480.shape
    print "mbe: ", mbe.shape

    label = np.zeros((mbe.shape[0], len(__class_labels)))
    tmp_data = np.array(desc_dict[audio_filename])

    # discretizzazione del tempo inizio
    frame_start = np.floor(tmp_data[:, 0] * sr / hop_len).astype(int)
    # discretizzazione del tempo fine
    frame_end = np.ceil(tmp_data[:, 1] * sr / hop_len).astype(int)

    se_class = tmp_data[:, 2].astype(int)
    for ind, val in enumerate(se_class):
        label[frame_start[ind]:frame_end[ind], val] = 1
    #MBE
    tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(
        audio_filename, 'mon' if is_mono else 'bin'))
    np.savez(tmp_feat_file, mbe, label)
    #Save GCC in different folders
    if not is_mono:
        #120
        if '120' in RESOLUTIONS:
            tmp_feat_file_GCC_120= os.path.join(feat_folder, '{}_{}_{}.npz'.format(
                audio_filename, 'mon' if is_mono else 'bin', 'GCC_120'))
            np.savez(tmp_feat_file_GCC_120, GCC_120, label)
        #240
        if '240' in RESOLUTIONS:
            tmp_feat_file_GCC_240= os.path.join(feat_folder, '{}_{}_{}.npz'.format(
                audio_filename, 'mon' if is_mono else 'bin', 'GCC_240'))
            np.savez(tmp_feat_file_GCC_240, GCC_240, label)
        #480
        if '480' in RESOLUTIONS:
            tmp_feat_file_GCC_480= os.path.join(feat_folder, '{}_{}_{}.npz'.format(
                audio_filename, 'mon' if is_mono else 'bin', 'GCC_480'))
            np.savez(tmp_feat_file_GCC_480, GCC_480, label)

# -----------------------------------------------------------------------
# Feature Normalization
# -----------------------------------------------------------------------

for fold in folds_list:
    train_file = os.path.join(evaluation_setup_folder,
                              'street_fold{}_train.txt'.format(1))
    evaluate_file = os.path.join(
        evaluation_setup_folder, 'street_fold{}_evaluate.txt'.format(1))
    train_dict = load_desc_file(train_file)
    test_dict = load_desc_file(evaluate_file)

    # mbe
    X_train, Y_train, X_test, Y_test = None, None, None, None
    #gcc
    if not is_mono:
        # 120
        if '120' in RESOLUTIONS:
            X_train_GCC_120, Y_train_GCC_120, X_test_GCC_120, Y_test_GCC_120 = None, None, None, None
        # 240
        if '240' in RESOLUTIONS:
            X_train_GCC_240, Y_train_GCC_240 ,X_test_GCC_240, Y_test_GCC_240 = None, None, None, None
        # 480
        if '480' in RESOLUTIONS:
            X_train_GCC_480, Y_train_GCC_480 ,X_test_GCC_480, Y_test_GCC_480 = None, None, None, None
    
    for key in train_dict.keys():
        # mbe
        tmp_feat_file = os.path.join(
            feat_folder, '{}_{}.npz'.format(key, 'mon' if is_mono else 'bin'))
        dmp = np.load(tmp_feat_file)
        tmp_mbe, tmp_label = dmp['arr_0'], dmp['arr_1']
        #gcc
        if not is_mono:
            # 120
            if '120' in RESOLUTIONS:
                tmp_feat_file_GCC_120= os.path.join(
                    feat_folder, '{}_{}_{}.npz'.format(key, 'mon' if is_mono else 'bin', 'GCC_120'))
                dmp = np.load(tmp_feat_file_GCC_120)
                tmp_GCC_120, tmp_label = dmp['arr_0'], dmp['arr_1']
                if tmp_feat_file_GCC_120.endswith('.npz') and fold == max(folds_list) :
                    print('Delete: ', tmp_feat_file_GCC_120)
                    os.unlink(tmp_feat_file_GCC_120)
            #240
            if '240' in RESOLUTIONS:
                tmp_feat_file_GCC_240= os.path.join(
                    feat_folder, '{}_{}_{}.npz'.format(key, 'mon' if is_mono else 'bin', 'GCC_240'))
                dmp = np.load(tmp_feat_file_GCC_240)
                tmp_GCC_240, tmp_label = dmp['arr_0'], dmp['arr_1']
                if tmp_feat_file_GCC_240.endswith('.npz') and fold == max(folds_list) :
                    print('Delete: ', tmp_feat_file_GCC_240)
                    os.unlink(tmp_feat_file_GCC_240)
            #480
            if '480' in RESOLUTIONS:
                tmp_feat_file_GCC_480= os.path.join(
                    feat_folder, '{}_{}_{}.npz'.format(key, 'mon' if is_mono else 'bin', 'GCC_480'))
                dmp = np.load(tmp_feat_file_GCC_480)
                tmp_GCC_480, tmp_label = dmp['arr_0'], dmp['arr_1']
                if tmp_feat_file_GCC_480.endswith('.npz') and fold == max(folds_list) :
                    print('Delete: ', tmp_feat_file_GCC_480)
                    os.unlink(tmp_feat_file_GCC_480)


        if X_train is None:
            # mbe
            X_train, Y_train = tmp_mbe, tmp_label
            if not is_mono:
                # 120
                if '120' in RESOLUTIONS:
                    X_train_GCC_120, Y_train_GCC_120 = tmp_GCC_120, tmp_label
                # 240
                if '240' in RESOLUTIONS:
                    X_train_GCC_240, Y_train_GCC_240 = tmp_GCC_240, tmp_label
                # 480
                if '480' in RESOLUTIONS:
                    X_train_GCC_480, Y_train_GCC_480 = tmp_GCC_480, tmp_label
        else:
            # mbe
            X_train, Y_train = np.concatenate(
                (X_train, tmp_mbe), 0), np.concatenate((Y_train, tmp_label), 0)
            if not is_mono:
                # 120
                if '120' in RESOLUTIONS:
                    X_train_GCC_120, Y_train_GCC_120 = np.concatenate(
                        (X_train_GCC_120, tmp_GCC_120), 0), np.concatenate((Y_train_GCC_120, tmp_label), 0)
                # 240
                if '240' in RESOLUTIONS:
                    X_train_GCC_240, Y_train_GCC_240 = np.concatenate(
                        (X_train_GCC_240, tmp_GCC_240), 0), np.concatenate((Y_train_GCC_240, tmp_label), 0)
                # 480
                if '480' in RESOLUTIONS:
                    X_train_GCC_480, Y_train_GCC_480 = np.concatenate(
                        (X_train_GCC_480, tmp_GCC_480), 0), np.concatenate((Y_train_GCC_480, tmp_label), 0)

    for key in test_dict.keys():
        # mbe
        tmp_feat_file = os.path.join(
            feat_folder, '{}_{}.npz'.format(key, 'mon' if is_mono else 'bin'))
        dmp = np.load(tmp_feat_file)
        tmp_mbe, tmp_label = dmp['arr_0'], dmp['arr_1']
        if not is_mono:
            # 120
            if '120' in RESOLUTIONS:
                tmp_feat_file_GCC_120= os.path.join(
                    feat_folder, '{}_{}_{}.npz'.format(key, 'mon' if is_mono else 'bin', 'GCC_120'))
                dmp = np.load(tmp_feat_file_GCC_120)
                tmp_GCC_120, tmp_label = dmp['arr_0'], dmp['arr_1']
            #240
            if '240' in RESOLUTIONS:
                tmp_feat_file_GCC_240= os.path.join(
                    feat_folder, '{}_{}_{}.npz'.format(key, 'mon' if is_mono else 'bin', 'GCC_240'))
                dmp = np.load(tmp_feat_file_GCC_240)
                tmp_GCC_240, tmp_label = dmp['arr_0'], dmp['arr_1']
            #480
            if '480' in RESOLUTIONS:
                tmp_feat_file_GCC_480= os.path.join(
                    feat_folder, '{}_{}_{}.npz'.format(key, 'mon' if is_mono else 'bin', 'GCC_480'))
                dmp = np.load(tmp_feat_file_GCC_480)
                tmp_GCC_480, tmp_label = dmp['arr_0'], dmp['arr_1']


        if X_test is None:
            # mbe
            X_test, Y_test = tmp_mbe, tmp_label
            if not is_mono:
                # 120
                if '120' in RESOLUTIONS:
                    X_test_GCC_120, Y_test_GCC_120 = tmp_GCC_120, tmp_label
                # 240
                if '240' in RESOLUTIONS:
                    X_test_GCC_240, Y_test_GCC_240 = tmp_GCC_240, tmp_label
                # 480
                if '480' in RESOLUTIONS:
                    X_test_GCC_480, Y_test_GCC_480 = tmp_GCC_480, tmp_label
        else:
            # mbe
            X_test, Y_test = np.concatenate(
                (X_test, tmp_mbe), 0), np.concatenate((Y_test, tmp_label), 0)
            if not is_mono:
                # 120
                if '120' in RESOLUTIONS:
                    X_test_GCC_120, Y_test_GCC_120 = np.concatenate(
                        (X_test_GCC_120, tmp_GCC_120), 0), np.concatenate((Y_test_GCC_120, tmp_label), 0)
                # 240
                if '240' in RESOLUTIONS:
                    X_test_GCC_240, Y_test_GCC_240 = np.concatenate(
                        (X_test_GCC_240, tmp_GCC_240), 0), np.concatenate((Y_test_GCC_240, tmp_label), 0)
                # 480
                if '480' in RESOLUTIONS:
                    X_test_GCC_480, Y_test_GCC_480 = np.concatenate(
                        (X_test_GCC_480, tmp_GCC_480), 0), np.concatenate((Y_test_GCC_480, tmp_label), 0)

    # Normalize the training data, and scale the testing data using the training data weights
    scaler = preprocessing.StandardScaler()
    # mbe
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    normalized_feat_file = os.path.join(
        feat_folder, 'mbe_{}_fold{}.npz'.format('mon' if is_mono else 'bin', fold))
    np.savez(normalized_feat_file, X_train, Y_train, X_test, Y_test)
    print('normalized_feat_file : {}'.format(normalized_feat_file))
    if not is_mono:
        # GCC
        #120
        if '120' in RESOLUTIONS:
            normalized_feat_file_GCC_120 = os.path.join(
                feat_folder, 'GCC_120_{}_fold{}.npz'.format('mon' if is_mono else 'bin', fold))
            np.savez(normalized_feat_file_GCC_120, X_train_GCC_120,
                    Y_train_GCC_120, X_test_GCC_120, Y_test_GCC_120)
            print('normalized_feat_file_GCC_120 : {}'.format(normalized_feat_file_GCC_120))

        #240
        if '240' in RESOLUTIONS:
            normalized_feat_file_GCC_240 = os.path.join(
                feat_folder, 'GCC_240_{}_fold{}.npz'.format('mon' if is_mono else 'bin', fold))
            np.savez(normalized_feat_file_GCC_240, X_train_GCC_240,
                    Y_train_GCC_240, X_test_GCC_240, Y_test_GCC_240)
            print('normalized_feat_file_GCC_240 : {}'.format(normalized_feat_file_GCC_240))

        #480
        if '480' in RESOLUTIONS:
            normalized_feat_file_GCC_480 = os.path.join(
                feat_folder, 'GCC_480_{}_fold{}.npz'.format('mon' if is_mono else 'bin', fold))
            np.savez(normalized_feat_file_GCC_480, X_train_GCC_480,
                    Y_train_GCC_480, X_test_GCC_480, Y_test_GCC_480)
            print('normalized_feat_file_GCC_480 : {}'.format(normalized_feat_file_GCC_480))
