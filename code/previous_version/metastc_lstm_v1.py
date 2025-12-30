
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ==========================================
# Device Configuration
# Priority: CUDA -> MPS -> CPU
# ==========================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# ==========================================
# 1. Traffic2Vec (Temporal Encoder)
# Paper Section III.A - Temporal Features (Eq. 1, 2, 3)
# ==========================================
class Traffic2Vec(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Traffic2Vec, self).__init__()
        # input_dim should cover traffic flow + normalized time values
        
        # Learnable parameters for Eq. 2
        # w_d: daily frequency, w_w: weekly frequency
        # Initialization logic approximates 2pi/24 and 2pi/(24*7) but allows fine-tuning
        self.w_d = nn.Parameter(torch.tensor(2 * math.pi / 24.0)) 
        self.w_w = nn.Parameter(torch.tensor(2 * math.pi / (24.0 * 7.0)))
        
        self.phi_1 = nn.Parameter(torch.randn(1))
        self.phi_2 = nn.Parameter(torch.randn(1))
        
        self.w_0 = nn.Parameter(torch.randn(1))
        self.b_0 = nn.Parameter(torch.zeros(1))
        
        # Dense layer for Eq. 3
        # Input: Original features + 3 encoding terms (linear, sin_day, sin_week)
        self.dense = nn.Linear(input_dim + 3, hidden_dim)

    def forward(self, x, t_norm):
        """
        x: Traffic flow sequence [Batch, Seq_Len, Feature_Dim]
        t_norm: Normalized time steps [Batch, Seq_Len, 1] (e.g., hour of day)
        """
        # Eq. 2: Traffic2Vec(t) components
        linear_term = self.w_0 * t_norm + self.b_0
        daily_term = torch.sin(self.w_d * t_norm + self.phi_1)
        weekly_term = torch.sin(self.w_w * t_norm + self.phi_2)
        
        # Concatenate original input with encodings
        # Shape: [Batch, Seq_Len, Feature_Dim + 3]
        combined = torch.cat([x, linear_term, daily_term, weekly_term], dim=-1)
        
        # Eq. 3: h_t = Dense(...)
        h_t = F.relu(self.dense(combined))
        return h_t

# ==========================================
# 2. Spatial Encoder
# Paper Section III.A - Spatial Features (Eq. 4)
# ==========================================
class SpatialEncoder(nn.Module):
    def __init__(self, num_roads, num_levels, num_lanes, embed_dim):
        super(SpatialEncoder, self).__init__()
        
        # Embeddings for discrete features
        self.embed_road = nn.Embedding(num_roads, embed_dim)
        self.embed_level = nn.Embedding(num_levels, embed_dim)
        self.embed_lane = nn.Embedding(num_lanes, embed_dim)
        
        # Linear projection to match temporal dimension if needed
        # Resulting dimension will be 3 * embed_dim
        self.out_dim = 3 * embed_dim

    def forward(self, road_idx, road_level, lane_num):
        """
        Inputs are integer indices [Batch]
        """
        # Eq. 4: Concatenation of embeddings
        emb_r = self.embed_road(road_idx)
        emb_l = self.embed_level(road_level)
        emb_n = self.embed_lane(lane_num)
        
        # Concatenate: [Batch, 3 * embed_dim]
        h_s = torch.cat([emb_r, emb_l, emb_n], dim=-1)
        return h_s

# ==========================================
# 3. Spatio-Temporal Gating Unit
# Paper Section III.A - Gating (Eq. 5, 6, 7)
# ==========================================
class SpatioTemporalGating(nn.Module):
    def __init__(self, spatial_dim, temporal_dim):
        super(SpatioTemporalGating, self).__init__()
        
        # Eq. 5: G_s weights
        self.W_ss = nn.Linear(spatial_dim, spatial_dim)
        self.W_ts = nn.Linear(temporal_dim, spatial_dim)
        
        # Eq. 6: G_t weights
        self.W_st = nn.Linear(spatial_dim, temporal_dim)
        self.W_tt = nn.Linear(temporal_dim, temporal_dim)
        
        self.final_dim = spatial_dim + temporal_dim

    def forward(self, h_s, h_t):
        """
        h_s: Spatial representation [Batch, Spatial_Dim]
        h_t: Temporal representation [Batch, Seq_Len, Temporal_Dim]
        """
        # Expand h_s to match sequence length for calculation: [Batch, Seq_Len, Spatial_Dim]
        seq_len = h_t.size(1)
        h_s_expanded = h_s.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Eq. 5: G_s
        # sigmoid(W_ss * h_s + W_ts * h_t + b)
        g_s = torch.sigmoid(self.W_ss(h_s_expanded) + self.W_ts(h_t))
        
        # Eq. 6: G_t
        # sigmoid(W_st * h_s + W_tt * h_t + b)
        g_t = torch.sigmoid(self.W_st(h_s_expanded) + self.W_tt(h_t))
        
        # Eq. 7: h = [G_s * h_s || G_t * h_t]
        # Element-wise product then concatenate
        fused_s = g_s * h_s_expanded
        fused_t = g_t * h_t
        
        h = torch.cat([fused_s, fused_t], dim=-1)
        
        # For clustering, we often take the mean or last step representation
        # The paper implies h_j uses this representation. 
        # We return the sequence for the backbone, and a pooled vector for clustering.
        h_pooled = torch.mean(h, dim=1) 
        
        return h, h_pooled

# ==========================================
# 4. Cluster-Aware Memory Unit & Meta Learner
# Paper Section III.B (Eq. 8 - 12)
# ==========================================
class ClusterMemoryUnit(nn.Module):
    def __init__(self, hidden_dim, num_clusters, mem_dim):
        super(ClusterMemoryUnit, self).__init__()
        self.num_clusters = num_clusters
        self.mem_dim = mem_dim
        
        # Cluster Centroids (p_k in Eq. 8) - Trainable or updated via K-Means
        self.centroids = nn.Parameter(torch.randn(num_clusters, hidden_dim))
        
        # Memory Matrix (M_c) - Stores meta-parameters per cluster
        # Initialized randomly
        self.memory = nn.Parameter(torch.randn(num_clusters, mem_dim))
        
        # Gate weights for Eq. 9, 10, 11
        # Inputs to gates are (similarity * Memory), so dimension is mem_dim
        self.W_r = nn.Linear(mem_dim, mem_dim) # Reset gate
        self.W_u = nn.Linear(mem_dim, mem_dim) # Update gate
        self.W_c = nn.Linear(mem_dim, mem_dim) # Candidate memory

    def get_similarity(self, h_j):
        """
        Eq. 8: Compute similarity score a_j^k
        h_j: Hidden representation of road j [Batch, Hidden_Dim]
        """
        # Dot product between road hidden state and centroids
        # h_j: [B, H], centroids: [K, H] -> scores: [B, K]
        scores = torch.matmul(h_j, self.centroids.t())
        
        # Softmax to get probabilities/attention
        alpha = F.softmax(scores, dim=1)
        return alpha

    def forward(self, h_j, prev_M_c=None):
        """
        Retrieves and updates memory parameters based on road representation.
        """
        if prev_M_c is None:
            prev_M_c = self.memory # Use global memory if no local context yet
            
        # 1. Calculate Similarity (Eq. 8)
        alpha = self.get_similarity(h_j) # [Batch, K]
        
        # Retrieve Parameter Matrix M_{j,c} = a_j^k * M_c
        # Here we do a weighted sum of memory slots based on cluster similarity
        # [Batch, K] x [K, Mem_Dim] -> [Batch, Mem_Dim]
        M_j_c = torch.matmul(alpha, prev_M_c)
        
        # 2. Update Gates (Eq. 9, 10)
        # Note: The paper writes a_j^k \otimes M_{j,c}. Since M_{j,c} is already weighted,
        # we treat M_{j,c} as the input context for the gates.
        
        r_t = torch.sigmoid(self.W_r(M_j_c))
        u_t = torch.sigmoid(self.W_u(M_j_c))
        
        # 3. Candidate Memory (Eq. 11)
        # Usually LSTM uses (r_t * hidden), here applied to the retrieved memory
        c_tilde = torch.tanh(self.W_c(r_t * M_j_c))
        
        # 4. Update Memory (Eq. 12)
        # Note: This updates the "local" retrieval. 
        # In a full meta-learning loop, gradients would update the global self.memory
        current_M_c = (1 - u_t) * M_j_c + u_t * c_tilde
        
        return current_M_c, alpha

# ==========================================
# 5. Prediction Backbone (LSTM Example)
# Paper mentions "Backbone Agnostic".
# We use a Hypernetwork approach: Memory -> Weights for prediction head.
# ==========================================
class MetaBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, mem_dim):
        super(MetaBackbone, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Hypernetwork: Generate weights for the final linear layer using Memory
        # Input: Memory Context [Batch, Mem_Dim]
        # Output: Weights for Linear Layer [Batch, Hidden_Dim * Output_Dim] + Bias
        self.weight_gen = nn.Linear(mem_dim, hidden_dim * output_dim)
        self.bias_gen = nn.Linear(mem_dim, output_dim)
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x_seq, memory_context):
        """
        x_seq: Fused spatio-temporal sequence [Batch, Seq_Len, Feature_Dim]
        memory_context: Context vector from Memory Unit [Batch, Mem_Dim]
        """
        # Run standard LSTM
        lstm_out, _ = self.lstm(x_seq)
        # Take last time step for prediction
        last_hidden = lstm_out[:, -1, :] # [Batch, Hidden_Dim]
        
        # Generate Task-Specific Prediction Weights
        # This realizes "distinct parameters for the predictive layer"
        generated_weights = self.weight_gen(memory_context) # [Batch, H*O]
        generated_bias = self.bias_gen(memory_context)      # [Batch, O]
        
        # Reshape weights: [Batch, Hidden, Output]
        weights = generated_weights.view(-1, self.hidden_dim, self.output_dim)
        
        # Apply dynamic linear layer: y = xW + b
        # [Batch, 1, Hidden] bmm [Batch, Hidden, Output] -> [Batch, 1, Output]
        prediction = torch.bmm(last_hidden.unsqueeze(1), weights).squeeze(1) + generated_bias
        
        return prediction

# ==========================================
# 6. Main MetaSTC Framework
# Combines all modules
# ==========================================
class MetaSTC(nn.Module):
    def __init__(self, 
                 num_roads, num_levels, num_lanes,
                 input_flow_dim=1,
                 spatial_embed_dim=16,
                 temporal_hidden_dim=32,
                 num_clusters=5,
                 mem_dim=64,
                 output_seq_len=6):
        super(MetaSTC, self).__init__()
        
        # 1. Temporal Encoder
        self.traffic2vec = Traffic2Vec(input_dim=input_flow_dim, hidden_dim=temporal_hidden_dim)
        
        # 2. Spatial Encoder
        self.spatial_encoder = SpatialEncoder(num_roads, num_levels, num_lanes, spatial_embed_dim)
        spatial_out_dim = spatial_embed_dim * 3
        
        # 3. Gating
        self.gating = SpatioTemporalGating(spatial_dim=spatial_out_dim, temporal_dim=temporal_hidden_dim)
        
        # 4. Clustering & Memory
        fused_dim = spatial_out_dim + temporal_hidden_dim
        self.memory_unit = ClusterMemoryUnit(hidden_dim=fused_dim, num_clusters=num_clusters, mem_dim=mem_dim)
        
        # 5. Backbone (Predictor)
        self.backbone = MetaBackbone(input_dim=fused_dim, hidden_dim=64, output_dim=output_seq_len, mem_dim=mem_dim)

    def forward(self, x_flow, x_time, road_idx, road_level, lane_num):
        """
        Args:
            x_flow: Traffic flow [Batch, Seq_Len, 1]
            x_time: Normalized time info [Batch, Seq_Len, 1]
            road_idx: [Batch]
            road_level: [Batch]
            lane_num: [Batch]
        """
        # A. Encode Features
        # Temporal: [B, T, Temp_Dim]
        h_t = self.traffic2vec(x_flow, x_time) 
        # Spatial: [B, Spat_Dim]
        h_s = self.spatial_encoder(road_idx, road_level, lane_num)
        
        # B. Gating & Fusion
        # h_seq: [B, T, Fused_Dim], h_pooled: [B, Fused_Dim]
        h_seq, h_pooled = self.gating(h_s, h_t)
        
        # C. Meta-Learning / Memory Retrieval
        # Retrieve context parameters based on spatio-temporal similarity
        # memory_context: [B, Mem_Dim] (Used to initialize/modulate backbone)
        memory_context, cluster_attn = self.memory_unit(h_pooled)
        
        # D. Prediction
        # Backbone generates specific weights using memory_context and predicts
        prediction = self.backbone(h_seq, memory_context)
        
        return prediction, cluster_attn

# ==========================================
# 7. Data Loading for Real Data (修正完整版)
# ==========================================
class TrafficDataset(Dataset):
    def __init__(self, 
                 pkl_path, 
                 feature_path, 
                 look_back=12, 
                 look_forward=6,
                 time_range=None,      # 新增: 用于按时间划分 e.g., (start_idx, end_idx)
                 max_flow_override=None  # 新增: 用于确保验证/测试集使用训练集的max_flow
                ):
        # 1. --- 原始数据加载逻辑 (之前被省略的部分) ---
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Data file not found at {pkl_path}")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file not found at {feature_path}")
            
        with open(pkl_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        # 加载路网特征并建立索引
        feature_df = pd.read_csv(feature_path)
        csv_ids = set(feature_df['link_ID'].unique())
        
        # 过滤 PKL 中不在 CSV 里的路段
        valid_data = [d for d in raw_data if d['id'] in csv_ids]
        id_list = [d['id'] for d in valid_data]
        # 这里定义了 data_flow
        data_flow = np.array([d['flow'] for d in valid_data]) # [Num_Roads, Time_Steps]
        original_time_len = data_flow.shape[1] # 在切分前记录原始时间长度
        
        # 重新对齐 feature_df 的顺序与 id_list 一致
        feature_df = feature_df.set_index('link_ID').loc[id_list].reset_index()
        
        # --- 数据归一化与切分 ---
        if max_flow_override is not None:
            self.max_flow = max_flow_override
        else:
            # 如果是训练集，则从其时间范围内计算max_flow
            self.max_flow = data_flow[:, time_range[0]:time_range[1]].max() if time_range else data_flow.max()

        data_flow = data_flow / (self.max_flow + 1e-10)
        if time_range:
            data_flow = data_flow[:, time_range[0]:time_range[1]]
        
        # 建立映射
        level_categories = sorted(feature_df['Kind'].unique().tolist())
        self.level_map = {l: i for i, l in enumerate(level_categories)}
        
        lane_categories = sorted(feature_df['LaneNum'].unique().tolist())
        self.lane_map = {l: i for i, l in enumerate(lane_categories)}
        
        self.road_map = {rid: i for i, rid in enumerate(id_list)}
        
        self.num_roads = len(id_list)
        self.num_levels = len(level_categories)
        self.num_lanes = len(lane_categories)

        # 2. --- 优化后的 Tensor 转换逻辑 ---
        # 转换为 Tensor 并常驻内存 (Float32节省空间)
        # shape: [Num_Roads, Total_Time_Steps, 1]
        self.data_flow = torch.from_numpy(data_flow).float().unsqueeze(-1)
        
        # 预生成时间索引
        # 使用原始时间长度来确保时间编码在训练集和验证集之间的一致性
        full_time_indices = np.linspace(0, 1, original_time_len)
        if time_range:
            self.time_indices = torch.from_numpy(full_time_indices[time_range[0]:time_range[1]]).float().unsqueeze(-1)
        else:
            self.time_indices = torch.from_numpy(full_time_indices).float().unsqueeze(-1)
        
        # 静态特征 Tensor 化
        self.road_ids = torch.LongTensor([self.road_map[rid] for rid in id_list])
        self.levels = torch.LongTensor([self.level_map[feature_df.iloc[i]['Kind']] for i in range(len(id_list))])
        self.lanes = torch.LongTensor([self.lane_map[feature_df.iloc[i]['LaneNum']] for i in range(len(id_list))])
        
        self.look_back = look_back
        self.look_forward = look_forward
        self.time_len = self.data_flow.shape[1]
        self.samples_per_road = self.time_len - look_back - look_forward

        # 核心优化：只存储索引
        self.total_samples = self.num_roads * self.samples_per_road

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 通过索引反推：是哪条路？是哪个时间点？
        road_idx = idx // self.samples_per_road
        time_start = idx % self.samples_per_road
        
        # 动态切片
        x_f = self.data_flow[road_idx, time_start : time_start + self.look_back]
        x_t = self.time_indices[time_start : time_start + self.look_back]
        tgt = self.data_flow[road_idx, time_start + self.look_back : time_start + self.look_back + self.look_forward].squeeze(-1)
        
        # 获取静态特征
        r_id = self.road_ids[road_idx]
        lvl = self.levels[road_idx]
        ln = self.lanes[road_idx]
        
        return x_f, x_t, r_id, lvl, ln, tgt

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 32
    PKL_PATH = "data/traffic_flow/1/20230306/part-00000.pkl" 
    FEATURE_PATH = "data/link_feature.csv"
    
    # Load Real Data
    try:
        dataset = TrafficDataset(PKL_PATH, FEATURE_PATH)
        if device == torch.device("mps"):
            # MPS 设备不支持多线程数据加载
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        else:   
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"Successfully loaded {len(dataset)} samples from {PKL_PATH}")
        # 根据实际数据动态更新维度
        NUM_ROADS = dataset.num_roads
        NUM_LEVELS = dataset.num_levels
        NUM_LANES = dataset.num_lanes
    except Exception as e:
        print(f"Error loading data: {e}. Falling back to dummy data.")
        NUM_ROADS, NUM_LEVELS, NUM_LANES = 1000, 4, 8
        class DummyDataset(Dataset):
            def __len__(self): return BATCH_SIZE * 2
            def __getitem__(self, idx):
                return (torch.randn(12, 1), torch.rand(12, 1), 
                        torch.randint(0, NUM_ROADS, (1,)).item(), 
                        torch.randint(0, NUM_LEVELS, (1,)).item(), 
                        torch.randint(0, NUM_LANES, (1,)).item(), 
                        torch.randn(6))
        dataset = DummyDataset()
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # Model Initialization
    model = MetaSTC(
        num_roads=NUM_ROADS,
        num_levels=NUM_LEVELS,
        num_lanes=NUM_LANES,
        input_flow_dim=1,
        spatial_embed_dim=16,
        temporal_hidden_dim=32,
        num_clusters=5,
        mem_dim=64,
        output_seq_len=6 # Prediction Horizon P
    ).to(device)
    
    # Get one batch of real data
    batch = next(iter(dataloader))
    x_f, x_t, r_id, r_lvl, l_num, target = [item.to(device) for item in batch]
    
    # Forward Pass with Real Data
    prediction, cluster_attention = model(x_f, x_t, r_id, r_lvl, l_num)
    
    print("MetaSTC Implementation Check:")
    print(f"Input Flow Shape: {x_f.shape}")
    print(f"Prediction Shape: {prediction.shape} (Expected: [{BATCH_SIZE}, Output_Seq_Len])")
    print(f"Cluster Attention Shape: {cluster_attention.shape} (Soft assignments to K clusters)")
    
    criterion = nn.MSELoss()
    loss = criterion(prediction, target)
    print(f"Sample Loss: {loss.item()}")