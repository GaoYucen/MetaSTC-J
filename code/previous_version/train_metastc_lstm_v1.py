import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import os
import time
import pickle
import matplotlib.pyplot as plt

# ==========================================
# 0. 引入之前的类 (请确保这些类已定义)
# ==========================================
# 为了方便运行，请确保 TrafficDataset, MetaSTC 等类已经包含在文件中
# 或者使用 import 导入: from metastc_lstm import MetaSTC, TrafficDataset
from code.previous_version.metastc_lstm import MetaSTC, TrafficDataset, get_device

# ==========================================
# 1. 配置参数
# ==========================================
CONFIG = {
    "batch_size": 128,        # 你的A40/Mac内存较大，可以尝试大 Batch
    "epochs": 50,             # 训练轮数
    "lr": 0.001,              # 学习率 (论文推荐 0.001)
    "patience": 5,            # 早停耐心值
    "split_ratio": 0.8,       # 训练集占比 (按时间)
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "pkl_path": "data/traffic_flow/1/20230306/part-00000.pkl",
    "feature_path": "data/link_feature.csv",
    "max_batches": 20,        # [新增] 限制每个Epoch的最大Batch数，大幅提升速度 (类似meta-LSTM)
    "model_dir": "param/metastc_lstm"
}

# ==========================================
# 2. 准备数据
# ==========================================
print("正在加载数据集...")

# --- 按时间顺序划分数据集 (与 meta-LSTM.py 保持一致) ---
# 1. 先加载一次数据以获取总时间长度
with open(CONFIG["pkl_path"], 'rb') as f:
    # 加载一个路段的数据来获取总时间步长
    time_len = len(pickle.load(f)[0]['flow'])

# 2. 计算切分点
train_end_idx = int(time_len * CONFIG["split_ratio"])

# 3. 创建训练集
print("创建训练集...")
train_dataset = TrafficDataset(
    CONFIG["pkl_path"], 
    CONFIG["feature_path"],
    time_range=(0, train_end_idx)
)
max_flow_for_norm = train_dataset.max_flow

# 4. 创建验证集，并传入训练集的max_flow以保证归一化一致
print("创建验证集...")
val_dataset = TrafficDataset(
    CONFIG["pkl_path"], 
    CONFIG["feature_path"],
    time_range=(train_end_idx, time_len),
    max_flow_override=max_flow_for_norm
)

print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")

os.makedirs(CONFIG["model_dir"], exist_ok=True)

# Mac MPS 上建议 num_workers=0，否则可能变慢或报错
train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

# ==========================================
# 3. 初始化模型
# ==========================================
device = get_device()
model = MetaSTC(
    num_roads=train_dataset.num_roads,
    num_levels=train_dataset.num_levels,
    num_lanes=train_dataset.num_lanes,
    input_flow_dim=1,
    spatial_embed_dim=16,
    temporal_hidden_dim=32,
    num_clusters=5,
    mem_dim=64,
    output_seq_len=6
).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

# ==========================================
# 4. 训练循环
# ==========================================
print(f"开始训练，使用设备: {device}")
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(CONFIG["epochs"]):
    start_time = time.time()
    
    # --- 训练阶段 ---
    model.train()
    total_train_loss = 0
    train_batches_done = 0
    
    for batch_idx, (x_f, x_t, r_id, r_lvl, ln, target) in enumerate(train_loader):
        # [新增] 快速迭代模式：达到最大Batch数后跳出
        if CONFIG["max_batches"] > 0 and batch_idx >= CONFIG["max_batches"]:
            break

        # 移动数据到设备
        x_f, x_t = x_f.to(device), x_t.to(device)
        r_id, r_lvl, ln = r_id.to(device), r_lvl.to(device), ln.to(device)
        target = target.to(device)
        
        # 1. 清空梯度
        optimizer.zero_grad()
        
        # 2. 前向传播
        # 注意：model 返回 (prediction, cluster_attention)
        prediction, _ = model(x_f, x_t, r_id, r_lvl, ln)
        
        # 3. 计算损失
        loss = criterion(prediction, target)
        
        # 4. 反向传播与更新
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        train_batches_done += 1
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}", end='\r')
            
    avg_train_loss = total_train_loss / train_batches_done
    
    # --- 验证阶段 (修改版：增加真实误差计算) ---
    model.eval()
    total_val_loss = 0
    total_real_mae = 0  # 新增：真实 MAE
    total_real_rmse = 0 # 新增：真实 RMSE
    
    # 获取最大流量值用于反归一化
    # 注意：full_dataset.max_flow 是 numpy 类型，转为 tensor 或 float
    max_flow = train_dataset.max_flow 
    
    with torch.no_grad():
        for batch_idx, (x_f, x_t, r_id, r_lvl, ln, target) in enumerate(val_loader):
            # [新增] 验证集也进行采样，加快评估速度
            if CONFIG["max_batches"] > 0 and batch_idx >= CONFIG["max_batches"] // 2:
                break

            x_f, x_t = x_f.to(device), x_t.to(device)
            r_id, r_lvl, ln = r_id.to(device), r_lvl.to(device), ln.to(device)
            target = target.to(device)
            
            prediction, _ = model(x_f, x_t, r_id, r_lvl, ln)
            
            # 1. 计算归一化 Loss (用于早停和模型选择，保持不变)
            loss = criterion(prediction, target)
            total_val_loss += loss.item()
            
            # 2. 计算反归一化的真实指标 (用于人类观察)
            # 还原回真实速度 (km/h)
            real_pred = prediction * max_flow
            real_target = target * max_flow
            
            # 计算真实 MAE
            total_real_mae += torch.mean(torch.abs(real_pred - real_target)).item()
            # 计算真实 MSE 用于 RMSE
            total_real_rmse += torch.mean((real_pred - real_target) ** 2).item()
            
    # [修正] 计算平均值时使用实际运行的 batch 数量
    val_batches_done = (CONFIG["max_batches"] // 2) if CONFIG["max_batches"] > 0 else len(val_loader)
    avg_val_loss = total_val_loss / val_batches_done
    avg_real_mae = total_real_mae / val_batches_done
    avg_real_rmse = (total_real_rmse / val_batches_done) ** 0.5
    
    # --- 打印进度 (修改版) ---
    time_elapsed = time.time() - start_time
    print(f"\nEpoch {epoch+1}/{CONFIG['epochs']} | "
          f"Train Loss: {avg_train_loss:.6f} | "
          f"Val Loss (Norm): {avg_val_loss:.6f} | "
          f"Real MAE: {avg_real_mae:.4f} | "  # 这里会显示类似 3.57 的值
          f"Real RMSE: {avg_real_rmse:.4f} | " # 这里会显示类似 5.32 的值
          f"Time: {time_elapsed:.2f}s")
    
    # --- 模型保存与早停 ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(CONFIG["model_dir"], "best_metastc_model.pth"))
        print("  >>> 模型优化，已保存 checkpoint")
    else:
        patience_counter += 1
        print(f"  >>> Loss未下降，Patience: {patience_counter}/{CONFIG['patience']}")
        
    if patience_counter >= CONFIG["patience"]:
        print("早停触发 (Early Stopping)！训练结束。")
        break

print(f"训练完成！最佳模型已保存至 {CONFIG['model_dir']}/best_metastc_model.pth")

# ==========================================
# 5. 测试阶段 (与 meta-LSTM 指标对齐)
# ==========================================
print("\n开始最终测试评估...")
model.load_state_dict(torch.load(os.path.join(CONFIG["model_dir"], "best_metastc_model.pth"), weights_only=True))
model.eval()

all_preds = []
all_targets = []
max_flow = train_dataset.max_flow

with torch.no_grad():
    for x_f, x_t, r_id, r_lvl, ln, target in val_loader:
        x_f, x_t = x_f.to(device), x_t.to(device)
        r_id, r_lvl, ln = r_id.to(device), r_lvl.to(device), ln.to(device)
        
        prediction, _ = model(x_f, x_t, r_id, r_lvl, ln)
        
        # 反归一化
        real_pred = (prediction * max_flow).cpu().numpy()
        real_target = (target * max_flow).cpu().numpy()
        
        all_preds.append(real_pred)
        all_targets.append(real_target)

y_pred = np.concatenate(all_preds, axis=0).flatten()
y_true = np.concatenate(all_targets, axis=0).flatten()

def mean_absolute_percentage_error(y_true, y_pred):
    # 过滤掉真实值为0的点，防止除零错误
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

test_mae = mean_absolute_error(y_true, y_pred)
test_mse = mean_squared_error(y_true, y_pred)
test_rmse = np.sqrt(test_mse)
test_mape = mean_absolute_percentage_error(y_true, y_pred)
test_r2 = r2_score(y_true, y_pred)

print("-" * 30)
print(f"测试集评估结果 (基于反归一化真实值):")
print(f"Test MAE:   {test_mae:.4f}")
print(f"Test RMSE:  {test_rmse:.4f}")
print(f"Test MSE:   {test_mse:.4f}")
print(f"Test MAPE:  {test_mape:.4f}")
print(f"Test R2:    {test_r2:.4f}")
print("-" * 30)

# ==========================================
# 6. 绘制预测曲线 (与 meta-LSTM 风格保持一致)
# ==========================================
os.makedirs('figure', exist_ok=True)
plt.figure(figsize=(15, 6))
plt.grid(True, linestyle='--', alpha=0.5)

# 确定展示的时间步长度 (展示约24小时的数据)
look_forward = 6
display_steps = min(len(y_true) // look_forward, 288)

# 抽取预测结果 (取每个窗口的第一步预测以形成连续曲线)
indices = np.arange(0, display_steps * look_forward, look_forward)

if len(y_pred) >= indices[-1]:
    plt.plot(y_true[indices], label='Actual Speed (Ground Truth)', color='#4C72B0', linewidth=2, alpha=0.8)
    plt.plot(y_pred[indices], label='Predicted Speed (MetaSTC-LSTM)', color='#C44E52', linewidth=1.5, linestyle='--')
    
    plt.title(f'Traffic Speed Prediction Comparison (MetaSTC Test Set)', fontsize=14)
    plt.xlabel('Time Steps (5-min intervals)', fontsize=12)
    plt.ylabel('Speed (km/h)', fontsize=12)
    plt.legend(loc='upper right', frameon=True)
    
    # 限制纵轴范围
    y_min = np.min(y_true[indices]) * 0.8
    y_max = np.max(y_true[indices]) * 1.2
    plt.ylim(max(0, y_min), y_max)

plt.tight_layout()
save_path = 'figure/metastc_lstm_prediction.png'
plt.savefig(save_path, dpi=300)
print(f"预测曲线图已保存至 {save_path}")
plt.show()