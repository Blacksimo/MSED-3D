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

#---------------------------average---------------------------------------------------
average_f1 = []
average_er = []
for f in [1,2,3]:
    data_avg = np.load('colab/2D/sed_dropout=0.2/{}_story.npz'.format(f))
    er_overall_1sec_list_avg=data_avg['arr_0']
    f1_overall_1sec_list_avg=data_avg['arr_1']

    average_f1.append(f1_overall_1sec_list_avg[np.argmax(f1_overall_1sec_list_avg)])
    average_er.append(er_overall_1sec_list_avg[np.argmin(er_overall_1sec_list_avg)])

print average_f1
print 'best f1 mean :', np.mean(average_f1)
print 'best er mean :', np.mean(average_er)
#--------------------------------------------------------------------------------------------

#colab/3D/mbe_dropout=0.2

data = np.load('colab/2D/sed_dropout=0.2/2_story.npz')

er_overall_1sec_list=data['arr_0']
f1_overall_1sec_list=data['arr_1']
val_loss=data['arr_2']
tr_loss=data['arr_3'] #for the min er

print('f1 best_index er:', f1_overall_1sec_list[np.argmin(er_overall_1sec_list)],np.argmin(er_overall_1sec_list))
print('np.argmax(f1_overall_1sec_list):', f1_overall_1sec_list[np.argmax(f1_overall_1sec_list)],np.argmax(f1_overall_1sec_list))

early =(len(f1_overall_1sec_list) + np.argmax(f1_overall_1sec_list))/ 2

_nb_epoch = early

plt.figure()

plt.subplot(211)
plt.plot(range(_nb_epoch), er_overall_1sec_list[0:early], label='ER ')
#plt.plot(range(_nb_epoch), f1_overall_1sec_list[0:early], label='f1')
#plt.xlabel('epoch')
plt.ylabel('error rate')
plt.legend()
plt.grid(True)

plt.subplot(212)
plt.plot(range(_nb_epoch), f1_overall_1sec_list[0:early], label='f1')
plt.axvline(np.argmax(f1_overall_1sec_list), color='r', linestyle='--', label= "%.3f" % f1_overall_1sec_list[np.argmax(f1_overall_1sec_list)])
plt.axvline(np.argmin(f1_overall_1sec_list), color='g', linestyle=':', label= "%.3f" % f1_overall_1sec_list[np.argmin(f1_overall_1sec_list)])
plt.xlabel('epoch')
plt.ylabel('f1 score')
plt.legend()
plt.grid(True)

plt.show()


