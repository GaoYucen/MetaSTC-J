import pickle
import pandas as pd
import numpy as np
import os

def check_data(pkl_path, feature_path):
    print("="*50)
    print("开始数据格式检查...")
    print("="*50)

    # 1. 检查 .pkl 文件
    if not os.path.exists(pkl_path):
        print(f"错误: 找不到文件 {pkl_path}")
        return

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"PKL 数据类型: {type(data)}")
    print(f"样本总数 (路段数): {len(data)}")
    
    if len(data) > 0:
        first_item = data[0]
        print(f"单个样本键名: {list(first_item.keys())}")
        flow_data = np.array(first_item['flow'])
        print(f"流量数据形状: {flow_data.shape} (预期应为 [288])")
        print(f"流量数据示例 (前5个): {flow_data[:5]}")

    # 2. 检查 .csv 文件
    if not os.path.exists(feature_path):
        print(f"错误: 找不到文件 {feature_path}")
        return

    feature_df = pd.read_csv(feature_path)
    print(f"\nCSV 特征表形状: {feature_df.shape}")
    print(f"CSV 列名: {list(feature_df.columns)}")
    
    # 检查模型需要的关键列
    target_cols = ['link_ID', 'Kind', 'LaneNum']
    for col in target_cols:
        if col in feature_df.columns:
            unique_count = feature_df[col].nunique()
            print(f"列 '{col}' 唯一值数量: {unique_count}")
            if col != 'link_ID':
                print(f"列 '{col}' 示例值: {feature_df[col].unique()[:5]}")
        else:
            print(f"警告: CSV 中缺少关键列 '{col}'")

    # 3. 检查 ID 匹配情况
    pkl_ids = set([d['id'] for d in data])
    csv_ids = set(feature_df['link_ID'].unique())
    intersection = pkl_ids.intersection(csv_ids)
    
    print(f"\nID 匹配统计:")
    print(f"PKL 中的唯一 ID 数: {len(pkl_ids)}")
    print(f"CSV 中的唯一 ID 数: {len(csv_ids)}")
    print(f"交集 ID 数 (可用于训练的路段): {len(intersection)}")
    print(f"匹配率: {len(intersection)/len(pkl_ids)*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    # 使用你代码中定义的路径
    PKL_PATH = "data/traffic_flow/1/20230306/part-00000.pkl"
    FEATURE_PATH = "data/link_feature.csv"
    check_data(PKL_PATH, FEATURE_PATH)