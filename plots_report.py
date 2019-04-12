from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
"""
0,er_overall_1sec_list,
1,f1_overall_1sec_list,
2,conf_mat_list,
3,best_index,
4,val_loss,
5,tr_loss,
6,er_mean_list
"""

#'colab/3D/gcc_mbe_dropout=0.2'
#'colab/3D/mbe_dropout=0.2'
#'tesla/4fold_0.5_0.0001_pool_error'
path_folder = 'colab/3D/mbe_dropout=0.5'
fold = 2

#---------------------------average---------------------------------------------------
average_f1 = []
average_er = []
for f in [1,2,3,4]:
    data_avg = np.load('{}/{}_story.npz'.format(path_folder,f))
    er_overall_1sec_list_avg=data_avg['arr_0']
    f1_overall_1sec_list_avg=data_avg['arr_1']
    best_index_avg=data_avg['arr_3']
    average_f1.append(f1_overall_1sec_list_avg[np.argmax(f1_overall_1sec_list_avg)])
    average_er.append(er_overall_1sec_list_avg[best_index_avg])

print average_f1
print 'best f1 mean :', np.mean(average_f1)
print 'best er mean :', np.mean(average_er)
#--------------------------------------------------------------------------------------------

data = np.load('{}/{}_story.npz'.format(path_folder,fold))


er_overall_1sec_list=data['arr_0']
f1_overall_1sec_list=data['arr_1']
conf_mat_list=data['arr_2']
best_index=data['arr_3'] #for the min er
val_loss=data['arr_4']
tr_loss=data['arr_5']
er_mean_list=data['arr_6']
print('best_index:', f1_overall_1sec_list[best_index],best_index)
print('np.argmax(f1_overall_1sec_list):', f1_overall_1sec_list[np.argmax(f1_overall_1sec_list)],np.argmax(f1_overall_1sec_list))

early =(len(f1_overall_1sec_list) + np.argmax(f1_overall_1sec_list))/ 2

_nb_epoch = early

plt.figure()

plt.subplot(211)
plt.plot(range(_nb_epoch), er_overall_1sec_list[0:early], label='ER ')
#plt.plot(range(_nb_epoch), f1_overall_1sec_list[0:early], label='f1')
plt.plot(range(_nb_epoch), er_mean_list[0:early], label='mean ER')
#plt.xlabel('epoch')
plt.ylabel('error rate')
plt.legend()
plt.grid(True)

plt.subplot(212)
plt.plot(range(_nb_epoch), f1_overall_1sec_list[0:early], label='f1')
plt.axvline(np.argmax(f1_overall_1sec_list), color='r', linestyle='--', label= "%.3f" % f1_overall_1sec_list[np.argmax(f1_overall_1sec_list)])
plt.axvline(best_index, color='g', linestyle=':', label= "%.3f" % f1_overall_1sec_list[best_index])
plt.xlabel('epoch')
plt.ylabel('f1 score')
plt.legend()
plt.grid(True)

plt.show()


