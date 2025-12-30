import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import nn, optim
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time

import argparse

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# from sklearn.cluster import KMeans
# import copy
# import re
import time
import random
import copy  # For deep copying models

import sys
sys.path.append('')

look_back = 12
look_forward = 6

k = 5 # 恢复正常聚类数量

#%%
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(look_back * hidden_dim, look_forward * output_dim)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output.reshape(output.size(0), -1)
        output = self.linear(output)
        output = output.reshape(output.size(0), look_forward, output_dim)
        return output

class MAML:
    def __init__(self, model, lr_inner=0.01, lr_outer=0.001, n_inner_loop=1):
        self.model = model
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.n_inner_loop = n_inner_loop
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr_outer)

    def inner_loop(self, data, targets):
        with torch.enable_grad():
            for step in range(self.n_inner_loop):
                preds = self.model(data)
                loss = nn.MSELoss()(preds, targets)
                self.model.zero_grad()
                # 使用一阶近似 (FOMAML)，去掉 create_graph=True 以大幅提升速度
                grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
                with torch.no_grad():
                    for param, grad in zip(self.model.parameters(), grads):
                        param.data -= self.lr_inner * grad
        # This is an in-place update, so we don't need to return weights.

    def outer_loop(self, data, targets):
        self.inner_loop(data, targets)
        preds = self.model(data)
        loss = nn.MSELoss()(preds, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets, sequence_length):
        self.data = data
        self.targets = targets
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):
        return (self.data[index:index+self.sequence_length], self.targets[index+self.sequence_length])

from collections import Counter

# 为flatten函数创建一个辅助函数
def flatten(list_of_lists):
    """将嵌套列表扁平化为单一列表"""
    return [item for sublist in list_of_lists for item in sublist]

#%% 读取dict数据格式
def load_data(filename):
    import pickle
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # 确定data['flow']长度为288的id列表
    id_list = [d['id'] for d in data]

    # 将data中的'flow'抽取为新的numpy数组
    data_flow = np.array([d['flow'] for d in data])

    return data_flow, id_list

#%% Add argument parser at the beginning of your script
parser = argparse.ArgumentParser(description='MetaSTC model for traffic flow prediction')
parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test',
                    help='Mode of operation: train or test (default: train)')
parser.add_argument('--model_path', type=str, default=None,
                    help='Path to the saved model for testing mode')
args = parser.parse_args()

#%%
# 选择device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs('param/meta', exist_ok=True)
os.makedirs('log', exist_ok=True)
os.makedirs('figure', exist_ok=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print('Using {} device'.format(device))

#%% 加载数据并准备训练
data_flow, id_list = load_data('data/traffic_flow/1/20230306/part-00000.pkl')

#%% 读取feature
feature = pd.read_csv('data/link_feature.csv')

#%% 确定节点数和时间步数量
# 删除id_list不在feature['link_ID']中的条目，并删除data_flow对应的row
data_flow = data_flow[[i for i,x in enumerate(id_list) if x in feature['link_ID'].values]]
id_list = [i for i in id_list if i in feature['link_ID'].values]

# [修改] 归一化基准：仅使用训练集(前80%)的最大值，保持与 train_metastc_lstm.py 一致
train_size_for_norm = int(data_flow.shape[1] * 0.8)
max_value_flow = np.max(data_flow[:, :train_size_for_norm])
data_flow = data_flow / max_value_flow

time_len = data_flow.shape[1]
num_nodes = data_flow.shape[0]

#%%

# plit data_flow into training and testing sets
train_ratio = 0.8
train_size = int(time_len * train_ratio)
train_data = data_flow[:, :train_size]
test_data = data_flow[:, train_size:]
# Transpose train_data and test_data
train_data = np.transpose(train_data, (1, 0))
test_data = np.transpose(test_data, (1, 0))

#%%
# Split train_data into train_x and train_y
# Use look_back steps to predict the next look_forward steps
train_x, train_y = [], []

for i in range(train_size - look_back - look_forward):
    train_x.append(np.array(train_data[i:i+look_back, :]))
    train_y.append(np.array(train_data[i+look_back:i+look_back+look_forward, :]))
train_x = np.array(train_x)
train_y = np.array(train_y)

test_x, test_y = [], []

for i in range(time_len - train_size - look_back - look_forward):
    test_x.append(np.array(test_data[i:i+look_back, :]))
    test_y.append(np.array(test_data[i+look_back:i+look_back+look_forward, :]))
test_x = np.array(test_x)
test_y = np.array(test_y)

#%%
train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).float()
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).float()

#%% 根据id_list筛选feature['link_ID']对应的row组成新的feature_road
feature_road = feature[feature['link_ID'].isin(id_list)]

# 将feature_road按照id_list的顺序进行排序
feature_road = feature_road.set_index('link_ID')
feature_road = feature_road.loc[id_list]
feature_road = feature_road.reset_index()

# feature去掉link_ID列和geometry列
# feature_used = feature_road.drop(columns=['link_ID', 'geometry', 'Kind'])
feature_used = feature_road.drop(columns=['link_ID', 'Kind'])

# Drop rows with NaN values in feature_used
feature_used = feature_used.dropna(axis=1)

# Define a function to implement K-Means algorithm
def kmeans(data, k):
    """
    :param data: 2D numpy array
    :param k: number of clusters
    :return: labels of each data point
    """
    # Initialize centroids randomly
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    # Initialize labels as zeros
    labels = np.zeros(data.shape[0])
    # Iterate until convergence
    while True:
        # Calculate distances between each data point and each centroid
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        # Assign each data point to the closest centroid
        new_labels = np.argmin(distances, axis=0)
        # Check if labels have changed
        if np.array_equal(new_labels, labels):
            break
        # Update labels and centroids
        labels = new_labels
        for i in range(k):
            centroids[i] = np.mean(data[labels == i], axis=0)
    return labels

# Use kmeans function to cluster feature_used into 3 clusters
feature_used = np.array(feature_used.astype('float64'))
# 对feature_used的每一列进行归一化
for i in range(feature_used.shape[1]):
    max_value = np.max(feature_used[:, i])
    feature_used[:, i] = feature_used[:, i] / (max_value + 1e-10)
# 删除含nan的列
feature_used = feature_used[:, ~np.isnan(feature_used).any(axis=0)]
# 对data_flow的每一行的每12列进行求和平均
data_flow_new = np.zeros((data_flow.shape[0], 12))
# for i in range(data_flow.shape[0]):
#     for j in range(12):
#         data_flow_new[i, j] = np.mean(data_flow[i, j*look_back:(j+1)*look_back])
data_flow_new = np.stack([
    np.mean(data_flow[:, j*look_back:(j+1)*look_back], axis=1) for j in range(12)
]).T
# 将feature_uesd和data_flow进行拼接
feature_used_new = np.concatenate((feature_used, data_flow_new), axis=1)
feature_used_label = kmeans(feature_used_new, k)
#%% 存储feature_used_label到.txt文件
np.savetxt('param/meta/feature_used_label_'+str(look_forward)+'_'+str(k)+'.txt', feature_used_label)

#%%
# 读取feature_used_label
feature_used_label = np.loadtxt('param/meta/feature_used_label_'+str(look_forward)+'_'+str(k)+'.txt')
# 统计各项出现的次数
feature_used_label_count = np.zeros(k)
for i in range(feature_used_label.shape[0]):
    feature_used_label_count[int(feature_used_label[i])] += 1
print(feature_used_label_count)

#%%
def generate_train_x(k):
    train_x_ = {}
    for i in range(k):
        train_x_['train_x_'+str(i)] = []  # 这里可以替换为你的数据生成逻辑
    return train_x_

def generate_train_y(k):
    train_y_ = {}
    for i in range(k):
        train_y_['train_y_'+str(i)] = []  # 这里可以替换为你的数据生成逻辑
    return train_y_

train_x_ = generate_train_x(k)
train_y_ = generate_train_y(k)

#%%
for i in range(feature_used_label.shape[0]):
    for j in range(k):
        if feature_used_label[i] == j:
            train_x_['train_x_'+str(j)].append(train_x[:,:,i])
            train_y_['train_y_'+str(j)].append(train_y[:,:,i])

#%%
for i in range(k):
    if len(train_x_['train_x_'+str(i)]) > 0:
        train_x_['train_x_'+str(i)] = torch.stack(train_x_['train_x_'+str(i)])
    if len(train_y_['train_y_'+str(i)]) > 0:
        train_y_['train_y_'+str(i)] = torch.stack(train_y_['train_y_'+str(i)])

#%%
for i in range(k):
    train_x_['train_x_'+str(i)] = train_x_['train_x_'+str(i)].view(-1, look_back).unsqueeze(-1)
    train_y_['train_y_'+str(i)] = train_y_['train_y_'+str(i)].view(-1, look_forward).unsqueeze(-1)


#%%
def generate_test_x(k):
    test_x_ = {}
    for i in range(k):
        test_x_['test_x_'+str(i)] = []  # 这里可以替换为你的数据生成逻辑
    return test_x_

def generate_test_y(k):
    test_y_ = {}
    for i in range(k):
        test_y_['test_y_'+str(i)] = []  # 这里可以替换为你的数据生成逻辑
    return test_y_

test_x_ = generate_test_x(k)
test_y_ = generate_test_y(k)

#%%
for i in range(feature_used_label.shape[0]):
    for j in range(k):
        if feature_used_label[i] == j:
            test_x_['test_x_'+str(j)].append(test_x[:,:,i])
            test_y_['test_y_'+str(j)].append(test_y[:,:,i])

#%%
for i in range(k):
    if len(test_x_['test_x_'+str(i)]) > 0:
        test_x_['test_x_'+str(i)] = torch.stack(test_x_['test_x_'+str(i)])
    if len(test_y_['test_y_'+str(i)]) > 0:
        test_y_['test_y_'+str(i)] = torch.stack(test_y_['test_y_'+str(i)])

#%%
for i in range(k):
    test_x_['test_x_'+str(i)] = test_x_['test_x_'+str(i)].view(-1, look_back).unsqueeze(-1)
    test_y_['test_y_'+str(i)] = test_y_['test_y_'+str(i)].view(-1, look_forward).unsqueeze(-1)

#%%
# 创建训练集和测试集的TensorDataset
train_dataset_ = {}
test_dataset_ = {}
for i in range(k):
    train_dataset_['train_dataset_'+str(i)] = TensorDataset(train_x_['train_x_'+str(i)], train_y_['train_y_'+str(i)])
    test_dataset_['test_dataset_'+str(i)] = TensorDataset(test_x_['test_x_'+str(i)], test_y_['test_y_'+str(i)])

#%%
# 定义一个批处理大小
batch_size = 128

# 创建训练集和测试集的DataLoader
train_loader_ = {}
test_loader_ = {}
for i in range(k):
    train_loader_['train_loader_'+str(i)] = DataLoader(train_dataset_['train_dataset_'+str(i)], batch_size=batch_size, shuffle=True)
    test_loader_['test_loader_'+str(i)] = DataLoader(test_dataset_['test_dataset_'+str(i)], batch_size=batch_size, shuffle=False)

#%%
# 定义参数
input_dim = 1
hidden_dim = 20
output_dim = 1

# 初始化模型
model = LSTMModel(input_dim, hidden_dim, output_dim).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters in LSTM Model: {total_params}')

# model = torch.compile(model)  # Significantly faster execution
maml = MAML(model)

#%% # Where your training and testing logic currently is, modify to use the mode
if args.mode == 'train':
    # Training logic
    start_time = time.time()
    num_epochs = 50 # 设定正常的训练轮数
    max_batches_per_cluster = 20 # 每轮每个聚类只跑 20 个 batch，加快 epoch 迭代

    # Train model
    with open('log/train_log.txt', 'w') as f_train:
        f_train.write('Epoch,Loss,Time\n')
        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            count = 0
            for i in range(k):
                for j, (data, targets) in enumerate(train_loader_['train_loader_' + str(i)]):
                    data, targets = data.to(device), targets.to(device)
                    loss_val = maml.outer_loop(data, targets)
                    epoch_loss += loss_val
                    count += 1
                    if j >= max_batches_per_cluster - 1:
                        break
            avg_loss = epoch_loss / count if count > 0 else 0.0
            duration = time.time() - epoch_start
            print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} completed in {duration:.2f}s')
            f_train.write(f'{epoch+1},{avg_loss:.4f},{duration:.2f}\n')

    # Save model parameters before fine-tuning
    torch.save(maml.model.state_dict(), 'param/meta/maml_model_' + str(look_forward) + '.pth')

    # Fine-tuning for each cluster
    for i in range(k):
        maml.model.load_state_dict(torch.load('param/meta/maml_model_' + str(look_forward) + '.pth', weights_only=True))
        for data, targets in train_loader_['train_loader_' + str(i)]:
            data, targets = data.to(device), targets.to(device)
            maml.inner_loop(data, targets)
        torch.save(maml.model.state_dict(), 'param/meta/maml_model_' + str(look_forward) + '_' + str(i) + '.pth')

    end_time = time.time()
    print(f'Training completed in {end_time - start_time:.4f}s')

#%% 测试模型，输出测试集的准确率
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true.flatten()), np.array(y_pred.flatten())
    # Delete elements in y_pred that correspond to the deleted elements in y_true
    y_pred = np.delete(y_pred, np.where(y_true == 0))
    # Delete elements in y_true that are equal to 0
    y_true = y_true[y_true != 0]
    return np.mean(np.abs((y_true - y_pred) / y_true))

#%% 读取模型参数
start_time = time.time()

# 统一使用标准推理模式，移除 RL 调试
with torch.inference_mode():
    preds_list = []
    targets_list = []

    for i in range(k):
        model_file = args.model_path if args.model_path else f'param/meta/maml_model_{look_forward}_{i}.pth'
        maml.model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))

        for data, targets in test_loader_['test_loader_'+str(i)]:
            data, targets = data.to(device), targets.to(device)
            preds = model(data)
            preds = preds * max_value_flow
            targets = targets * max_value_flow
            preds = preds.cpu().detach().numpy().flatten()
            targets = targets.cpu().detach().numpy().flatten()
            preds_list = np.append(preds_list, preds)
            targets_list = np.append(targets_list, targets)

end_time = time.time()

print('Test Time: {:.4f}s'.format(end_time - start_time))
print('Test MAE: {:.4f}'.format(mean_absolute_error(preds_list, targets_list)))
print('Test RMSE: {:.4f}'.format(np.sqrt(mean_squared_error(preds_list, targets_list))))
print('Test MSE: {:.4f}'.format(mean_squared_error(preds_list, targets_list)))
print('Test MAPE: {:.4f}'.format(mean_absolute_percentage_error(preds_list, targets_list)))
print('Test R2: {:.4f}'.format(r2_score(targets_list, preds_list)))

#%% 将结果打印到log/meta-LSTM.txt文件中
f = open('log/meta-LSTM.txt', 'a')
f.write('look_back: {:.4f}s\n'.format(look_back))
f.write('Time: {:.4f}s\n'.format(end_time - start_time))
f.write('Test MAE: {:.4f}\n'.format(mean_absolute_error(preds_list, targets_list)))
f.write('Test RMSE: {:.4f}\n'.format(np.sqrt(mean_squared_error(preds_list, targets_list))))
f.write('Test MSE: {:.4f}\n'.format(mean_squared_error(preds_list, targets_list)))
f.write('Test MAPE: {:.4f}\n'.format(mean_absolute_percentage_error(preds_list, targets_list)))
f.write('Test R2: {:.4f}\n'.format(r2_score(targets_list, preds_list)))

#%% 绘制前1000个真实值和预测值
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))

# 设置绘图风格
plt.grid(True, linestyle='--', alpha=0.5)

# [修改] 绘图逻辑：精准定位原始数据集中的第 0 号路段，确保与 train_metastc_lstm.py 统一
num_test_samples_per_road = time_len - train_size - look_back - look_forward
display_steps = min(num_test_samples_per_road, 288)

# 1. 找到第 0 号路段所属的聚类
road0_cluster = int(feature_used_label[0])
# 2. 计算在 targets_list 中的偏移量 (跳过前面所有聚类的路段数据)
plot_offset = 0
for c in range(road0_cluster):
    num_roads_in_c = np.sum(feature_used_label == c)
    plot_offset += num_roads_in_c * num_test_samples_per_road * look_forward

# 3. 提取该路段的连续数据
indices = np.arange(0, display_steps * look_forward, look_forward)

if len(targets_list) >= plot_offset + indices[-1]:
    plt.plot(targets_list[plot_offset + indices], label='Actual Speed (Ground Truth)', color='#4C72B0', linewidth=2, alpha=0.8)
    plt.plot(preds_list[plot_offset + indices], label='Predicted Speed (Meta-LSTM)', color='#C44E52', linewidth=1.5, linestyle='--')
    
    plt.title(f'Traffic Speed Prediction Comparison (Test Set Sample)', fontsize=14)
    plt.xlabel('Time Steps (5-min intervals)', fontsize=12)
    plt.ylabel('Speed (km/h)', fontsize=12)
    plt.legend(loc='upper right', frameon=True)
    
    # 限制纵轴范围以更好地观察波动
    y_min = np.min(targets_list[indices]) * 0.8
    y_max = np.max(targets_list[indices]) * 1.2
    plt.ylim(max(0, y_min), y_max)

plt.tight_layout()
plt.savefig('figure/meta_lstm_prediction.png', dpi=300)
print("预测曲线图已保存至 figure/meta_lstm_prediction.png")
plt.show()
