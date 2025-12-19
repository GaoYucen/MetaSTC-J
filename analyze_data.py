import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # For Chinese characters if needed
plt.rcParams['axes.unicode_minus'] = False

def load_traffic_data(pkl_path):
    print(f"Loading traffic data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Convert to DataFrame
    # data is list of dicts: [{'id': ..., 'flow': array}, ...]
    ids = [item['id'] for item in data]
    flows = [item['flow'] for item in data]
    
    df_flow = pd.DataFrame(flows, index=ids)
    df_flow.index.name = 'link_ID'
    return df_flow

def load_link_features(csv_path):
    print(f"Loading link features from {csv_path}...")
    df_features = pd.read_csv(csv_path)
    # Ensure link_ID is string or int consistent with pickle
    # Pickle IDs looked like ints. CSV IDs look like ints.
    return df_features

def analyze_temporal_patterns():
    base_dir = "/Users/gaoyucen/Library/Mobile Documents/com~apple~CloudDocs/2. 论文/期刊1. ICDM 24转期刊/MetaSTC-J/data"
    pkl_path = os.path.join(base_dir, "traffic_flow/1/20230306/part-00000.pkl")
    csv_path = os.path.join(base_dir, "link_feature.csv")
    
    # 1. Load Data
    df_flow = load_traffic_data(pkl_path)
    df_features = load_link_features(csv_path)
    
    print(f"Traffic Data Shape: {df_flow.shape}")
    print(f"Link Features Shape: {df_features.shape}")
    
    # 2. Merge Data
    # Filter features to only those in flow data
    common_ids = df_flow.index.intersection(df_features['link_ID'])
    print(f"Number of links with both flow and features: {len(common_ids)}")
    
    df_features_filtered = df_features[df_features['link_ID'].isin(common_ids)].set_index('link_ID')
    df_flow_filtered = df_flow.loc[common_ids]
    
    # 3. Global Temporal Pattern
    mean_flow = df_flow_filtered.mean(axis=0)
    std_flow = df_flow_filtered.std(axis=0)
    
    # Time axis (288 steps = 24 hours * 12 steps/hour)
    time_steps = np.arange(288)
    hours = time_steps / 12.0
    
    # 4. Pattern by Road Level (FuncClass)
    # FuncClass: 1 (Highway) to 5 (Local) usually
    road_levels = df_features_filtered['FuncClass'].unique()
    road_levels.sort()
    
    level_means = {}
    for level in road_levels:
        ids_level = df_features_filtered[df_features_filtered['FuncClass'] == level].index
        if len(ids_level) > 0:
            level_means[level] = df_flow_filtered.loc[ids_level].mean(axis=0)
            
    # ================= PLOTTING =================
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Global Mean Flow with Std Dev
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(hours, mean_flow, label='Mean Flow', color='blue', linewidth=2)
    ax1.fill_between(hours, mean_flow - std_flow, mean_flow + std_flow, color='blue', alpha=0.1, label='Std Dev')
    ax1.set_title('Global Average Traffic Flow (24 Hours)')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Flow Volume')
    ax1.set_xticks(np.arange(0, 25, 2))
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Mean Flow by Road Level
    ax2 = plt.subplot(2, 2, 2)
    colors = plt.cm.viridis(np.linspace(0, 1, len(level_means)))
    for i, (level, series) in enumerate(level_means.items()):
        ax2.plot(hours, series, label=f'Level {level}', color=colors[i], linewidth=1.5)
    ax2.set_title('Average Traffic Flow by Road Level')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Flow Volume')
    ax2.set_xticks(np.arange(0, 25, 2))
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Heatmap of Top 50 Links (by total volume)
    ax3 = plt.subplot(2, 1, 2)
    # Calculate total volume for each link
    total_volume = df_flow_filtered.sum(axis=1)
    top_50_ids = total_volume.nlargest(50).index
    top_50_flow = df_flow_filtered.loc[top_50_ids]
    
    # Normalize for heatmap visualization (Min-Max scaling per link to show pattern)
    # Or just raw values? Let's use raw values but log scale might be better if variance is huge.
    # Let's stick to raw values first.
    sns.heatmap(top_50_flow, ax=ax3, cmap='viridis', cbar_kws={'label': 'Flow'})
    ax3.set_title('Traffic Flow Heatmap (Top 50 High-Volume Links)')
    ax3.set_xlabel('Time Step (5-min intervals)')
    ax3.set_ylabel('Link ID')
    # Adjust x-ticks to show hours
    xticks = np.arange(0, 289, 24) # Every 2 hours (24 steps)
    xticklabels = [f"{int(x/12)}:00" for x in xticks]
    ax3.set_xticks(xticks)
    ax3.set_xticklabels(xticklabels, rotation=0)
    
    plt.tight_layout()
    plt.savefig('traffic_pattern_analysis.png', dpi=300)
    print("Analysis plot saved to traffic_pattern_analysis.png")
    
    # 5. Clustering Analysis (Characterizing Temporal Patterns)
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    print("\nRunning K-Means Clustering on Temporal Patterns...")
    # Normalize rows (links) to focus on shape of pattern, not absolute magnitude
    scaler = StandardScaler()
    # We want to cluster based on the time series shape. 
    # Option A: Normalize each time series to mean 0 std 1.
    # Option B: Use raw values.
    # Let's use Option A to find "patterns" regardless of speed limit.
    X = df_flow_filtered.values
    X_norm = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-5)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_norm)
    
    df_flow_filtered['Cluster'] = labels
    
    # Plot 4: Cluster Centroids
    fig2 = plt.figure(figsize=(12, 6))
    colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF']
    
    for i in range(4):
        cluster_mean = df_flow_filtered[df_flow_filtered['Cluster'] == i].drop('Cluster', axis=1).mean(axis=0)
        count = sum(labels == i)
        plt.plot(hours, cluster_mean, label=f'Pattern {i+1} (n={count})', linewidth=2)
        
    plt.title('Identified Temporal Patterns (Clustered by Shape)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Speed (km/h)')
    plt.xticks(np.arange(0, 25, 2))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('traffic_clusters.png', dpi=300)
    print("Cluster plot saved to traffic_clusters.png")

if __name__ == "__main__":
    analyze_temporal_patterns()


if __name__ == "__main__":
    analyze_temporal_patterns()
