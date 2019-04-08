# coding=utf-8
from __future__ import print_function
import os
import numpy as np
import time
import sys
import matplotlib.pyplot as plot
from keras.layers import Bidirectional, TimeDistributed, Conv2D,Conv3D, MaxPooling2D,MaxPooling3D, Input, GRU, Dense, Activation, Dropout, Reshape, Permute, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.models import Model
import keras
from sklearn.metrics import confusion_matrix
import metrics
import utils
from IPython import embed
import keras.backend as K
K.set_image_data_format('channels_first')
plot.switch_backend('agg')
sys.setrecursionlimit(10000)
print(K.image_data_format())

#-----------------------------------------------
# Load data MBE
#-----------------------------------------------
def load_data(_feat_folder, _mono, _fold=None):
    feat_file_fold = os.path.join(_feat_folder, 'mbe_{}_fold{}.npz'.format('mon' if _mono else 'bin', _fold))
    dmp = np.load(feat_file_fold)
    _X_train, _Y_train, _X_test, _Y_test = dmp['arr_0'],  dmp['arr_1'],  dmp['arr_2'],  dmp['arr_3']
    #(9028, 80) --->  (9028, 40, 2)
    return _X_train, _Y_train, _X_test, _Y_test

#-----------------------------------------------
# Load data GCC
#-----------------------------------------------
def load_data_GCC(_feat_folder, _mono, _fold=None):
    #MODIFICATO PER IL GCC ESTRATTO DA feature_gcc.py nella cartella feat_gcc
    feat_file_fold_120 = os.path.join(_feat_folder, 'GCC_120_{}_fold{}.npz'.format('mon' if _mono else 'bin', _fold))
    dmp_120 = np.load(feat_file_fold_120)
    _X_train_120, _Y_train_120, _X_test_120, _Y_test_120 = dmp_120['arr_0'],  dmp_120['arr_1'],  dmp_120['arr_2'],  dmp_120['arr_3']
    print("_X_train_120: ", _X_train_120.shape)

    feat_file_fold_240 = os.path.join(_feat_folder, 'GCC_240_{}_fold{}.npz'.format('mon' if _mono else 'bin', _fold))
    dmp_240 = np.load(feat_file_fold_240)
    _X_train_240, _Y_train_240, _X_test_240, _Y_test_240 = dmp_240['arr_0'],  dmp_240['arr_1'],  dmp_240['arr_2'],  dmp_240['arr_3']

    feat_file_fold_480 = os.path.join(_feat_folder, 'GCC_480_{}_fold{}.npz'.format('mon' if _mono else 'bin', _fold))
    dmp_480 = np.load(feat_file_fold_480)
    _X_train_480, _Y_train_480, _X_test_480, _Y_test_480 = dmp_480['arr_0'],  dmp_480['arr_1'],  dmp_480['arr_2'],  dmp_480['arr_3']
    #TODO
    #conacatenate axis = 1--> 60x3 tau, vedi anche concatenazione canali mbe il concetto è lo stesso
    _X_train, _Y_train = np.concatenate(
        (_X_train_120, _X_train_240,_X_train_480), 1), np.concatenate((_Y_train_120, _Y_train_240,_Y_train_480), 0)
    
    _X_test, _Y_test = np.concatenate(
                (_X_test_120, _X_test_240,_X_test_480), 1), np.concatenate((_Y_test_120, _Y_test_240,_Y_test_480), 0)
    #shape risultante è shape= (time, 60x3)--> (time,180)

    print("_X_train: ", _X_train.shape)
    print("_X_test: ", _X_test.shape)
    print("_Y_train: ", _Y_train.shape)
    print("_Y_test: ", _Y_test.shape)
    return _X_train, _Y_train, _X_test, _Y_test




def get_model(data_in_mbe, data_in_gcc, data_out, _cnn_nb_filt, _cnn_pool_size_mbe,_cnn_pool_size_gcc, _rnn_nb, _fc_nb, _nb_ch, _gcc_ch):

    #----------------------------------------------------------------------------------------------------------------------
    # MBE branch
    #----------------------------------------------------------------------------------------------------------------------
    spec_start = Input(shape=(data_in_mbe.shape[-4], data_in_mbe.shape[-3], data_in_mbe.shape[-2], data_in_mbe.shape[-1]))
    spec_x = spec_start
    
    for _i, _cnt in enumerate(_cnn_pool_size_mbe):
        if _i == 0:
            spec_x = Conv3D(filters=_cnn_nb_filt, kernel_size=(_nb_ch, 3, 3), padding='same')(spec_x)
            spec_x = BatchNormalization(axis=1)(spec_x)
            spec_x = Activation('relu')(spec_x)
            spec_x = MaxPooling3D(pool_size=(1, 1 , _cnn_pool_size_mbe[_i]))(spec_x)
            spec_x = Dropout(dropout_rate)(spec_x)
            spec_x = Reshape((-1,  data_in_mbe.shape[-2], 8))(spec_x)
        else:
            spec_x = Conv2D(filters=_cnn_nb_filt, kernel_size=(3, 3), padding='same')(spec_x)
            spec_x = BatchNormalization(axis=1)(spec_x)
            spec_x = Activation('relu')(spec_x)
            spec_x = MaxPooling2D(pool_size=(1, _cnn_pool_size_mbe[_i]))(spec_x)
            spec_x = Dropout(dropout_rate)(spec_x)
    spec_x = Permute((2, 1, 3))(spec_x)
    spec_x = Reshape((data_in_mbe.shape[-2], -1))(spec_x)
    print("spec_x: ", spec_x.shape)


    #----------------------------------------------------------------------------------------------------------------------
    # GCC branch
    #----------------------------------------------------------------------------------------------------------------------
    spec_start_gcc = Input(shape=(data_in_gcc.shape[-4], data_in_gcc.shape[-3], data_in_gcc.shape[-2], data_in_gcc.shape[-1]))
    spec_x_gcc = spec_start_gcc
    
    for _i, _cnt in enumerate(_cnn_pool_size_gcc):
        if _i == 0:
            spec_x_gcc = Conv3D(filters=_cnn_nb_filt, kernel_size=(_gcc_ch, 3, 3), padding='same')(spec_x_gcc)
            spec_x_gcc = BatchNormalization(axis=1)(spec_x_gcc)
            spec_x_gcc = Activation('relu')(spec_x_gcc)
            spec_x_gcc = MaxPooling3D(pool_size=(1, 1 , _cnn_pool_size_mbe[_i]))(spec_x_gcc)
            spec_x_gcc = Dropout(dropout_rate)(spec_x_gcc)
            spec_x_gcc = Reshape((-1,  data_in_mbe.shape[-2], 8))(spec_x_gcc)
        else:
            spec_x_gcc = Conv2D(filters=_cnn_nb_filt, kernel_size=(3, 3), padding='same')(spec_x_gcc)
            spec_x_gcc = BatchNormalization(axis=1)(spec_x_gcc)
            spec_x_gcc = Activation('relu')(spec_x_gcc)
            spec_x_gcc = MaxPooling2D(pool_size=(1, _cnn_pool_size_mbe[_i]))(spec_x_gcc)
            spec_x_gcc = Dropout(dropout_rate)(spec_x_gcc)
    spec_x_gcc = Permute((2, 1, 3))(spec_x_gcc)
    spec_x_gcc = Reshape((data_in_gcc.shape[-2], -1))(spec_x_gcc)
    print("spec_x_gcc: ", spec_x_gcc.shape)

    #non va bene perchè non è piu un tensor
    #spec_conc = Concatenate([spec_x,spec_x_gcc])
    # così invece funziona
    spec_x_conc = Concatenate()([spec_x,spec_x_gcc])
    print("spec_conc: ", spec_x_conc.shape)
    #TODO       
    """
            concatenazione(spec_x - spec_x_gcc)
                          |
              LSTM fw           LSTM bw
                          |
             concatenazione(spec_x - spec_x_gcc)
                          |
               LSTM fw           LSTM bw
                          |
            concatenazione  (spec_x - spec_x_gcc)
                          |
               Fully connected sigmoid 
    """

    #----------------------------------------------------
    # RNN           
    #----------------------------------------------------
    
    for _r in _rnn_nb:
        spec_x_conc = Bidirectional(
            GRU(_r, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True),
            merge_mode ='concat')(spec_x_conc)
    
    """
    spec_x_conc = Bidirectional(GRU(_rnn_nb[0], activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,return_sequences= True, go_backwards= True),
            merge_mode ='concat')(spec_x_conc)
    print("spec_conc: ", spec_x_conc.shape)

    spec_x_conc = Bidirectional(GRU(_rnn_nb[1], activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,return_sequences= True,go_backwards= True),
            merge_mode ='concat')(spec_x_conc)
    """
    
    for _f in _fc_nb:
        spec_x_conc = TimeDistributed(Dense(_f))(spec_x_conc)
        spec_x_conc = Dropout(dropout_rate)(spec_x_conc)
 

    # Dense - out T x 6 CLASSES
    spec_x_conc = TimeDistributed(Dense(data_out.shape[-1]))(spec_x_conc)
    out = Activation('sigmoid', name='strong_out')(spec_x_conc)

    _model = Model(inputs=[spec_start, spec_start_gcc], outputs=out)
    _model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy']) #lr = 1x10-4
    _model.summary()
    return _model


def plot_functions(_nb_epoch, _tr_loss, _val_loss, _f1, _er, extension=''):
    plot.figure()

    plot.subplot(211)
    plot.plot(range(_nb_epoch), _tr_loss, label='train loss')
    plot.plot(range(_nb_epoch), _val_loss, label='val loss')
    plot.legend()
    plot.grid(True)

    plot.subplot(212)
    plot.plot(range(_nb_epoch), _f1, label='f')
    plot.plot(range(_nb_epoch), _er, label='er')
    plot.legend()
    plot.grid(True)

    plot.savefig(__models_dir + __fig_name + extension)
    plot.close()
    print('figure name : {}'.format(__fig_name))

#-----------------------------------------------------------------
# Preprocess data MBE
#-----------------------------------------------------------------
def preprocess_data(_X, _Y, _X_test, _Y_test, _seq_len, _nb_ch):
    # split into sequences
    _X = utils.split_in_seqs(_X, _seq_len)
    _Y = utils.split_in_seqs(_Y, _seq_len)

    _X_test = utils.split_in_seqs(_X_test, _seq_len)
    _Y_test = utils.split_in_seqs(_Y_test, _seq_len)

    _X = utils.split_multi_channels(_X, _nb_ch)
    _X_test = utils.split_multi_channels(_X_test, _nb_ch)
    return _X, _Y, _X_test, _Y_test

#-----------------------------------------------------------------
# Preprocess data GCC NON serve in teoria
#-----------------------------------------------------------------
def preprocess_data_GCC(_X, _Y, _X_test, _Y_test, _seq_len, _nb_ch):
    # split into sequences
    _X = utils.split_in_seqs(_X, _seq_len)
    _Y = utils.split_in_seqs(_Y, _seq_len)

    _X_test = utils.split_in_seqs(_X_test, _seq_len)
    _Y_test = utils.split_in_seqs(_Y_test, _seq_len)

    _X = utils.split_multi_channels_COMPLEX(_X, _nb_ch)
    _X_test = utils.split_multi_channels_COMPLEX(_X_test, _nb_ch)
    return _X, _Y, _X_test, _Y_test




class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self._er = 0
        self._er_list = []
        self._er_prev = 0
        self._fail_count=0
        self._f1 = 0
        self._f1_list = []
    def on_epoch_end(self,epoch, batch, logs={}):
        X_val,X_val_gcc, y_val = self.validation_data[0], self.validation_data[1],self.validation_data[2]
        # Calculate the predictions on test data, in order to calculate ER and F scores
        pred = model.predict([X_val,X_val_gcc])
        #È true o false
        pred_thresh = pred > 0.5  #0.5 threeshold vedi paper
        #print pred_thresh
        score_list = metrics.compute_scores(pred_thresh, y_val, frames_in_1_sec=frames_1_sec) #Y_TEST shape = (220,256,6)
        #score list è un dizionario
        self._f1 = score_list['f1_overall_1sec']
        self._er = score_list['er_overall_1sec']

        if  self._er > self._er_prev:
            self._fail_count+=1
            if self._fail_count >= 10:
                print('Epoch: ', epoch, ' Custom ER: ', self._er, ' Failcount: ', self._fail_count )
                self.model.stop_training = True
        else:
            #resetto il patience count
            self._fail_count = 0
        #aggiorno
	    print(' Prev: ', self._er_prev ,' Custom ER: ', self._er,' Failcount: ', self._fail_count )
        self._er_prev = self._er
        self._er_list.append(self._er)
        self._f1_list.append(self._f1)
        return

    def get_data(self):
        return self._er, self._er_list, self._f1,self._f1_list
   


#######################################################################################
# MAIN SCRIPT STARTS HERE
#######################################################################################

is_mono = False  # True: mono-channel input, False: binaural input

feat_folder = 'feat/'
__fig_name = '{}_{}'.format('mon' if is_mono else 'bin', time.strftime("%Y_%m_%d_%H_%M_%S"))


nb_ch = 1 if is_mono else 2
batch_size = 64   # Decrease this if you want to run on smaller GPU's
seq_len = 256       # Frame sequence length. Input to the CRNN.
nb_epoch = 1000      # Training epochs
patience = 100  # Patience for early stopping
#patience = int(0.25 * nb_epoch)  # Patience for early stopping
gcc_ch = 3 #gcc resolutions 

# Number of frames in 1 second, required to calculate F and ER for 1 sec segments.
# Make sure the nfft and sr are the same as in feature.py
sr = 44100
nfft = 2048
frames_1_sec = int(sr/(nfft/2.0))

print('\n\nUNIQUE ID: {}'.format(__fig_name))
print('TRAINING PARAMETERS: nb_ch: {},gcc_ch: {}, seq_len: {}, batch_size: {}, nb_epoch: {}, frames_1_sec: {}'.format(
    nb_ch,gcc_ch, seq_len, batch_size, nb_epoch, frames_1_sec))

# Folder for saving model and training curves
__models_dir = 'models/'
utils.create_folder(__models_dir)

# CRNN model definition
cnn_nb_filt = 64 #128            # CNN filter size
cnn_pool_size_mbe = [5, 2, 2]   # Maxpooling across frequency. Length of cnn_pool_size =  number of CNN layers
cnn_pool_size_gcc = [5, 3, 2]   # Maxpooling across frequency. Length of cnn_pool_size =  number of CNN layers
rnn_nb = [64, 64]   # Q in the paper         # Number of RNN nodes.  Length of rnn_nb =  number of RNN layers
fc_nb = [32]                # Number of FC nodes.  Length of fc_nb =  number of FC layers
dropout_rate = 0.2 #0.5        # Dropout after each layer
print('MODEL PARAMETERS:\n cnn_nb_filt: {}, cnn_pool_size_mbe: {}, cnn_pool_size_gcc: {}, rnn_nb, Q units: {}, fc_nb: {}, dropout_rate: {}'.format(
    cnn_nb_filt, cnn_pool_size_mbe, cnn_pool_size_gcc, rnn_nb, fc_nb, dropout_rate))

avg_er = list()
avg_f1 = list()

X_MBE, Y, X_test_MBE, Y_test = None, None,None,None
X_GCC, Y_GCC, X_test_GCC, Y_test_GCC = None, None,None,None

fold = 9 # per differenziare il modello salvato, unica fold
for fold_u in [1,2,3,4]:
    print('\n\n----------------------------------------------')
    print('FOLD: {}'.format(fold_u), ' Concatenation')
    print('----------------------------------------------\n')
    #--------------------------------------------------------
    # Load feature and labels, pre-process it
    #----------------------------------------------------------
    #MBE
    X_MBE_fld, Y_fld, X_test_MBE_fld, Y_test_fld = load_data(feat_folder, is_mono, fold_u)
    if X_MBE is None: #va bene uno qualunque che sia None
        X_MBE = X_MBE_fld
        Y = Y_fld
        X_test_MBE = X_test_MBE_fld
        Y_test = Y_test_fld
    else:
        X_MBE = np.concatenate((X_MBE, X_MBE_fld), 0)
        Y = np.concatenate((Y, Y_fld), 0)
        X_test_MBE = np.concatenate((X_test_MBE, X_test_MBE_fld), 0)
        Y_test = np.concatenate((Y_test, Y_test_fld), 0)

    #GCC
    #X_GCC_fld, Y_GCC_fld, X_test_GCC_fld, Y_test_GCC_fld = load_data_GCC(feat_folder, is_mono, fold_u)
    
    X_GCC_fld =np.zeros((726868,180),dtype=np.complex_)
    Y_GCC_fld =np.zeros((726868,6))
    X_test_GCC_fld =np.zeros((225512,180),dtype=np.complex_)
    Y_test_GCC_fld =np.zeros((225512,6))
    
    if X_GCC is None: #va bene uno qualunque che sia None
        X_GCC = X_GCC_fld
        Y_GCC = Y_GCC_fld
        X_test_GCC = X_test_GCC_fld
        Y_test_GCC = Y_test_GCC_fld
    else:
        X_GCC = np.concatenate((X_GCC, X_GCC_fld), 0)
        Y_GCC = np.concatenate((Y_GCC, Y_GCC_fld), 0)
        X_test_GCC = np.concatenate((X_test_GCC, X_test_GCC_fld), 0)
        Y_test_GCC = np.concatenate((Y_test_GCC, Y_test_GCC_fld), 0)


print("X_MBE shape FOLD: ", X_MBE.shape)
print("Y_test shape FOLD: ", Y_test.shape)
print("X_GCC shape FOLD: ", X_GCC.shape)


X_MBE, Y, X_test_MBE, Y_test = preprocess_data(X_MBE, Y, X_test_MBE, Y_test, seq_len, nb_ch)
print("X_MBE shape Preprocessed: ", X_MBE.shape)
print("Y_test shape Preprocessed: ", Y_test.shape)

X_GCC, Y_GCC, X_test_GCC, Y_test_GCC = preprocess_data_GCC(X_GCC, Y_GCC, X_test_GCC, Y_test_GCC, seq_len, gcc_ch)
print("X_GCC shape Preprocessed: ", X_GCC.shape)

#------------------------------------------------
# 3D Conv layer Reshape
#------------------------------------------------
#MBE
X_MBE = X_MBE.reshape(X_MBE.shape[-4] ,1, X_MBE.shape[-3], X_MBE.shape[-2], X_MBE.shape[-1])
X_test_MBE = X_test_MBE.reshape(X_test_MBE.shape[-4] ,1, X_test_MBE.shape[-3], X_test_MBE.shape[-2], X_test_MBE.shape[-1]) #(?,1,2,256,40)
print("X_MBE 3DConv: ", X_MBE.shape)
#GCC
X_GCC = X_GCC.reshape(X_GCC.shape[-4] ,1, X_GCC.shape[-3], X_GCC.shape[-2], X_GCC.shape[-1])
X_test_GCC = X_test_GCC.reshape(X_test_GCC.shape[-4] ,1, X_test_GCC.shape[-3], X_test_GCC.shape[-2], X_test_GCC.shape[-1]) #(?,1,3,256,60)
print("X_GCC 3DConv: ", X_GCC.shape)



#------------------------------------------
# Load model
#-------------------------------------------
#L'output è uno quindi è uguale Y o Y_MBE, uguale per Y_test
model = get_model(X_MBE, X_GCC, Y, cnn_nb_filt, cnn_pool_size_mbe,cnn_pool_size_gcc, rnn_nb, fc_nb, nb_ch, gcc_ch)

#-------------------------------------------
# Training
#-------------------------------------------
best_epoch, pat_cnt, best_er, f1_for_best_er, best_conf_mat = 0, 0, 99999, None, None
#tr_loss, val_loss, f1_overall_1sec_list, er_overall_1sec_list = [0] * nb_epoch, [0] * nb_epoch, [0] * nb_epoch, [0] * nb_epoch
posterior_thresh = 0.5
#early_stop = EarlyStopping(monitor='val_custom_loss', patience = 1, mode='min')
my_metrics = Metrics()

hist = model.fit(
        [X_MBE,X_GCC], Y,
        batch_size=batch_size,
        validation_data=[[X_test_MBE,X_test_GCC], Y_test],
        epochs=3,
        verbose=1,
        callbacks=[my_metrics]
    )
print('Training END')
last_er, er_overall_1sec_list, last_f1, f1_overall_1sec_list  = my_metrics.get_data()
val_loss = hist.history.get('val_loss')
tr_loss = hist.history.get('loss')

print('Saving model ...')
model.save(os.path.join(__models_dir, '{}_fold_{}_model.h5'.format(__fig_name, fold)))
print('Model saved!')

plot_functions(len(f1_overall_1sec_list), tr_loss, val_loss, f1_overall_1sec_list, er_overall_1sec_list, '_fold_{}'.format(fold))

print('')
print('tr Er : {}, val Er : {}, F1_overall : {}, ER_overall : {}'.format(
            tr_loss, val_loss, f1_overall_1sec_list[-1], er_overall_1sec_list[-1]))

