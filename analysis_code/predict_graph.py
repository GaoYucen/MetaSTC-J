#%% 读取npy数据
import numpy as np
import random

# 读取数据
data_true = np.load('STC-ML/data/FiLM_Epochs2/true_unorm.npy')
print(data_true.shape)
data_film = np.load('STC-ML/data/FiLM_Epochs2/pred_unorm.npy')
data_rf = np.load('STC-ML/data/RF_output.npy')

#%%读取csv数据
data_arima = np.loadtxt('STC-ML/data/Arima_Results/preds_recover.csv', delimiter=',')
# 转置
data_arima = data_arima.T

#%%
data_tgcn = np.loadtxt('STC-ML/data/tgcn_result.csv', delimiter=',')

#%% 
data_stc = np.load('STC-ML/data/MetaSTC_FiLM.npy')

#%%
print(data_true.flatten()[0:30])


#%% 画图
length = 30
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
# 加大字体
plt.rcParams['font.size'] = 16
# 不要加粗字体
plt.rcParams['font.weight'] = 'normal'
# 加粗线条
plt.rcParams['lines.linewidth'] = 2
# 每个线条加符号
plt.rcParams['lines.markersize'] = 7
markers = "dH^*ov"
# 画图
plt.plot(data_true.flatten()[0:length], label='True', color='grey', linewidth=4, marker=markers[5])
plt.plot(data_arima.flatten()[0:length], label='ARIMA', color='purple', marker=markers[4])
# plt.plot(data_rf.flatten()[0:length], label='RF', marker=markers[3])
plt.plot(data_tgcn.flatten()[0:length], label='T-GCN', marker=markers[2])
plt.plot(data_film.flatten()[0:length], label='FiLM', marker=markers[1])
plt.plot(data_stc.flatten()[0:length], label='MetaSTC+FiLM', color='red', linewidth=2, marker=markers[0])
plt.legend(loc='upper right')
# 添加y轴标签为m/s
# x轴坐标乘5，单位为min
plt.xticks(np.arange(0, length+1, 5), np.arange(0, (length+1)*5, 25))
plt.xlabel('Time Point (min)')
plt.ylabel('Traffic Flow (km/h)')
# plt.show()
plt.savefig('STC-ML/graph/MetaSTC_predict.pdf', bbox_inches='tight')
