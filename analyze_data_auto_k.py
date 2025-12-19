import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # For Chinese characters if needed
plt.rcParams['axes.unicode_minus'] = False

def load_traffic_data(pkl_path):
    print(f"Loading traffic data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Convert to DataFrame
    ids = [item['id'] for item in data]
    flows = [item['flow'] for item in data]
    
    df_flow = pd.DataFrame(flows, index=ids)
    df_flow.index.name = 'link_ID'
    return df_flow

def load_link_features(csv_path):
    print(f"Loading link features from {csv_path}...")
    df_features = pd.read_csv(csv_path)
    return df_features

def analyze_optimal_k():
    base_dir = "/Users/gaoyucen/Library/Mobile Documents/com~apple~CloudDocs/2. 论文/期刊1. ICDM 24转期刊/MetaSTC-J/data"
    pkl_path = os.path.join(base_dir, "traffic_flow/1/20230306/part-00000.pkl")
    csv_path = os.path.join(base_dir, "link_feature.csv")
    
    # 1. Load Data
    df_flow = load_traffic_data(pkl_path)
    df_features = load_link_features(csv_path)
    
    # 2. Merge Data
    common_ids = df_flow.index.intersection(df_features['link_ID'])
    df_flow_filtered = df_flow.loc[common_ids]
    print(f"Analyzing {len(df_flow_filtered)} links...")
    
    # 3. Prepare Data for Clustering
    print("\n=== Determining Optimal K (Automatic) ===")
    # Normalize rows (links) to focus on shape of pattern
    X = df_flow_filtered.values
    # Z-score normalization per time series (Mean=0, Std=1)
    # This ensures we cluster based on the *shape* of the daily pattern, not the absolute volume.
    X_norm = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-5)
    
    inertias = []
    silhouette_scores = []
    K_range = range(2, 10)  # Test K from 2 to 9
    
    # Downsample for silhouette score speed (use 3000 samples to be representative but fast)
    np.random.seed(42)
    sample_indices = np.random.choice(X_norm.shape[0], size=min(3000, X_norm.shape[0]), replace=False)
    X_sample = X_norm[sample_indices]
    
    print("Testing K values...")
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_norm)
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette score on sample
        labels_sample = kmeans.predict(X_sample)
        score = silhouette_score(X_sample, labels_sample)
        silhouette_scores.append(score)
        print(f"  K={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={score:.4f}")
        
    # --- Heuristic 1: Elbow Method (Max distance from line) ---
    p1 = np.array([K_range[0], inertias[0]])
    p2 = np.array([K_range[-1], inertias[-1]])
    max_dist = 0
    optimal_k_elbow = K_range[0]
    
    for i, k in enumerate(K_range):
        p = np.array([k, inertias[i]])
        # Perpendicular distance
        dist = np.abs(np.cross(p2-p1, p1-p)) / np.linalg.norm(p2-p1)
        if dist > max_dist:
            max_dist = dist
            optimal_k_elbow = k
            
    # --- Heuristic 2: Max Silhouette Score ---
    optimal_k_sil = K_range[np.argmax(silhouette_scores)]
    
    print(f"\nSuggested K (Elbow Method): {optimal_k_elbow}")
    print(f"Suggested K (Silhouette Score): {optimal_k_sil}")
    
    # 4. Plot K-Selection Metrics
    fig_k = plt.figure(figsize=(12, 5))
    
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(K_range, inertias, 'bo-', markersize=8)
    ax1.plot(optimal_k_elbow, inertias[K_range.index(optimal_k_elbow)], 'r*', markersize=15, label=f'Elbow (K={optimal_k_elbow})')
    ax1.set_title('Elbow Method (Inertia)')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia (SSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(K_range, silhouette_scores, 'go-', markersize=8)
    ax2.plot(optimal_k_sil, silhouette_scores[K_range.index(optimal_k_sil)], 'r*', markersize=15, label=f'Max Score (K={optimal_k_sil})')
    ax2.set_title('Silhouette Score')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimal_k_metrics.png', dpi=300)
    print("K-selection metrics plot saved to optimal_k_metrics.png")
    
    # 5. Generate Plots for Both Suggested K values
    k_options = [
        (optimal_k_sil, "Silhouette Score", "traffic_clusters_silhouette.png"),
        (optimal_k_elbow, "Elbow Method", "traffic_clusters_elbow.png")
    ]
    
    # Handle case where they are the same
    if optimal_k_sil == optimal_k_elbow:
        print(f"\nBoth methods suggest K={optimal_k_sil}. Generating one plot.")
        k_options = [(optimal_k_sil, "Both Methods", "traffic_clusters_auto.png")]

    for k, method_name, filename in k_options:
        print(f"\nGenerating plot for K={k} ({method_name})...")
        
        kmeans_final = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans_final.fit_predict(X_norm)
        df_flow_filtered['Cluster'] = labels
        
        fig = plt.figure(figsize=(12, 6))
        # Generate distinct colors
        colors = plt.cm.tab10(np.linspace(0, 1, k))
        
        # Time axis
        time_steps = np.arange(288)
        hours = time_steps / 12.0
        
        for i in range(k):
            # Calculate mean of the original (non-normalized) data for the cluster
            cluster_mean = df_flow_filtered[df_flow_filtered['Cluster'] == i].drop('Cluster', axis=1).mean(axis=0)
            count = sum(labels == i)
            percentage = (count / len(labels)) * 100
            plt.plot(hours, cluster_mean, label=f'Pattern {i+1} ({percentage:.1f}%)', color=colors[i], linewidth=2.5)
            
        plt.title(f'Identified Temporal Patterns (K={k} by {method_name})')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Speed (km/h)')
        plt.xticks(np.arange(0, 25, 2))
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(filename, dpi=300)
        print(f"Plot saved to {filename}")
        plt.close(fig)

if __name__ == "__main__":
    analyze_optimal_k()
