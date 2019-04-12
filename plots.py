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


data = np.load('colab/mbe_do_0.5/story_mbe/4_story.npz')
er_overall_1sec_list=data['arr_0']
f1_overall_1sec_list=data['arr_1']
conf_mat_list=data['arr_2']
best_index=data['arr_3'] #for the min er
val_loss=data['arr_4']
tr_loss=data['arr_5']
er_mean_list=data['arr_6']
print('best_index:', f1_overall_1sec_list[best_index-1])
print('np.argmax(f1_overall_1sec_list):', f1_overall_1sec_list[np.argmax(f1_overall_1sec_list)])

early = 400


mbe = data['arr_2']  # shape (7893,80)
time = np.arange(0, mbe.shape[0],1)

_nb_epoch = early

plt.figure()

plt.subplot(311)
plt.plot(range(_nb_epoch), er_overall_1sec_list[0:early], label='ER ')
#plt.plot(range(_nb_epoch), f1_overall_1sec_list[0:early], label='f1')
plt.plot(range(_nb_epoch), er_mean_list[0:early], label='mean ER')
plt.xlabel('epoch')
plt.ylabel('error rate')
plt.legend()
plt.grid(True)

plt.subplot(312)
plt.plot(range(_nb_epoch), val_loss[0:early], label='val_loss ')
plt.plot(range(_nb_epoch), tr_loss[0:early], label='train_loss')
plt.xlabel('epoch')
plt.ylabel('error rate')
plt.legend()
plt.grid(True)

plt.subplot(313)
plt.plot(range(_nb_epoch), f1_overall_1sec_list[0:early], label='f1')
plt.axvline(np.argmax(f1_overall_1sec_list), color='r', linestyle='--', label= '0.535')
plt.xlabel('epoch')
plt.ylabel('f1 score')
plt.legend()
plt.grid(True)

plt.show()


