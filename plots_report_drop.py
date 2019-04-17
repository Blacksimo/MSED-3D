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
#'colab/2D/mbe_dropout=0.5'
#'tesla/4fold_0.5_0.0001_pool_error'
#'tesla/gcc_mbe_dropout=0.5'


#data = np.load('{}/{}_story.npz'.format(path_folder,fold))
"""
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
"""

def load_plot_data(_folder_path):
    data = np.load(_folder_path)
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

    return er_overall_1sec_list, f1_overall_1sec_list, conf_mat_list, best_index, val_loss, tr_loss,er_mean_list,early
"""
_folder_path_05 =  'tesla/gcc_mbe_dropout=0.5/2_story.npz'
_folder_path_02 =  'tesla/gcc_mbe_dropout=0.2/2_story.npz'

"""
_folder_path_05 =  'colab/3D/mbe_dropout=0.5/4_story.npz'
_folder_path_02 =  'colab/3D/mbe_dropout=0.2/4_story.npz'
er_overall_1sec_list_2, f1_overall_1sec_list_2, conf_mat_list_2, best_index_2, val_loss_2, tr_loss,er_mean_list_2,early_2 = load_plot_data(_folder_path_02)
er_overall_1sec_list_5, f1_overall_1sec_list_5, conf_mat_list_5, best_index_5, val_loss_5, tr_loss,er_mean_list_5,early_5 = load_plot_data(_folder_path_05)

early = np.maximum(early_2,early_5)

plt.figure()

plt.subplot(211)
plt.plot(range(early), er_overall_1sec_list_2[0:early], label='ER .2')
plt.plot(range(early), er_mean_list_2[0:early], label='mean ER .2')

plt.plot(range(early), er_overall_1sec_list_5[0:early], label='ER .5 ')
plt.plot(range(early), er_mean_list_5[0:early], label='mean ER .5')


plt.ylabel('error rate')
plt.legend()
plt.grid(True)

plt.subplot(212)
plt.plot(range(early), f1_overall_1sec_list_2[0:early], label='f1')
plt.axvline(np.argmax(f1_overall_1sec_list_2), color='r', linestyle='--', label= "%.3f" % f1_overall_1sec_list_2[np.argmax(f1_overall_1sec_list_2)])
plt.axvline(best_index_2, color='r', linestyle=':', label= "%.3f" % f1_overall_1sec_list_2[best_index_2])


plt.plot(range(early), f1_overall_1sec_list_5[0:early], label='f1 .5')
plt.axvline(np.argmax(f1_overall_1sec_list_5), color='g', linestyle='--', label= "%.3f" % f1_overall_1sec_list_5[np.argmax(f1_overall_1sec_list_5)])
plt.axvline(best_index_5, color='g', linestyle=':', label= "%.3f" % f1_overall_1sec_list_5[best_index_5])

plt.xlabel('epoch')
plt.ylabel('f1 score')
plt.legend()
plt.grid(True)

plt.show()


