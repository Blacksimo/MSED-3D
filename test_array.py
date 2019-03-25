import numpy as np 
import os
is_mono = False 
feat_folder = 'feat_gcc/'
fold = 1


def load_data(_feat_folder, _mono, _fold=None):
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

    _X_train, _Y_train = np.concatenate(
        (_X_train_120, _X_train_240,_X_train_480), 1), np.concatenate((_Y_train_120, _Y_train_240,_Y_train_480), 0)
    
    _X_test, _Y_test = np.concatenate(
                (_X_test_120, _X_test_240,_X_test_480), 1), np.concatenate((_Y_test_120, _Y_test_240,_Y_test_480), 0)
    print("_X_train: ", _X_train.shape)
    print("_X_test: ", _X_test.shape)
    return _X_train, _Y_train, _X_test, _Y_test

X, Y, X_test, Y_test = load_data(feat_folder, is_mono, fold)

