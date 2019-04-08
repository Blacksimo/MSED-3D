import numpy as np
from sklearn.metrics import confusion_matrix
import metrics
import utils
import os
from sklearn import preprocessing
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
   
    return _X_train_120, _Y_train_120, _X_test_120, _Y_test_120, _X_train_240, _Y_train_240, _X_test_240, _Y_test_240,_X_train_480, _Y_train_480, _X_test_480, _Y_test_480


# input
feat_folder = 'feat_gcc_th/'
#output
out_folder = 'feat_gcc_norm/'
utils.create_folder(out_folder)
is_mono = False
for fold in [1, 2, 3, 4]:
    print('\n\n----------------------------------------------')
    print('FOLD: {}'.format(fold))
    print('----------------------------------------------\n')
    #--------------------------------------------------------
    #Normalize feature
    #----------------------------------------------------------

    #GCC
    X_train_120, Y_train_120, X_test_120, Y_test_120, X_train_240, Y_train_240, X_test_240, Y_test_240,X_train_480, Y_train_480, X_test_480, Y_test_480 = load_data_GCC(feat_folder, is_mono, fold)
    #Discard imaginary part
    #120
    X_train_120 = np.real(X_train_120)
    X_test_120 = np.real(X_test_120)
    #240
    X_train_240 = np.real(X_train_240)
    X_test_240 = np.real(X_test_240)
    #480
    X_train_480 = np.real(X_train_480)
    X_test_480 = np.real(X_test_480)
  
    
    # Normalize the training data, and scale the testing data using the training data weights
    #120
    print('Normalizing 120')
    scaler_120 = preprocessing.StandardScaler()
    X_train_120 = scaler_120.fit_transform(X_train_120)
    X_test_120 = scaler_120.transform(X_test_120)

    print('Saving 120')
    normalized_feat_file_GCC_120 = os.path.join(
                out_folder, 'GCC_120_{}_fold{}.npz'.format('mon' if is_mono else 'bin', fold))
    np.savez(normalized_feat_file_GCC_120, X_train_120,
                    Y_train_120, X_test_120, Y_test_120)
    print('normalized_feat_file_GCC_120 : {}'.format(normalized_feat_file_GCC_120))
    #240
    print('Normalizing 240')
    scaler_240 = preprocessing.StandardScaler()
    X_train_240 = scaler_240.fit_transform(X_train_240)
    X_test_240  = scaler_240.transform(X_test_240 )

    print('Saving 240')
    normalized_feat_file_GCC_240 = os.path.join(
                out_folder, 'GCC_240_{}_fold{}.npz'.format('mon' if is_mono else 'bin', fold))
    np.savez(normalized_feat_file_GCC_240, X_train_240,
                    Y_train_240, X_test_240, Y_test_240)
    print('normalized_feat_file_GCC_240 : {}'.format(normalized_feat_file_GCC_240))

    #480
    print('Normalizing 480')
    scaler_480 = preprocessing.StandardScaler()
    X_train_480 = scaler_480.fit_transform(X_train_480)
    X_test_480 = scaler_480.transform(X_test_480)
    print('Saving 480')

    normalized_feat_file_GCC_480 = os.path.join(
                out_folder, 'GCC_480_{}_fold{}.npz'.format('mon' if is_mono else 'bin', fold))
    np.savez(normalized_feat_file_GCC_480, X_train_480,
                    Y_train_480, X_test_480, Y_test_480)
    print('normalized_feat_file_GCC_480 : {}'.format(normalized_feat_file_GCC_480))



   
