import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os

def add_image(ax, img_path, x, y, zoom=0.5):
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)

def draw_box(ax, x, y, width, height, text, color='#E1F5FE', edge_color='#01579B', fontsize=10, icon_path=None):
    box = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.05", 
                                 linewidth=1.5, edgecolor=edge_color, facecolor=color)
    ax.add_patch(box)
    
    # If icon is present, shift text down
    text_y_offset = 0
    if icon_path:
        add_image(ax, icon_path, x + width/2, y + height/2 + 0.15, zoom=0.35)
        text_y_offset = -0.15
        
    ax.text(x + width/2, y + height/2 + text_y_offset, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color='#333333')
    return x + width/2, y + height/2, x + width, y + height/2, x, y + height/2, x + width/2, y

def draw_arrow(ax, x1, y1, x2, y2, color='#555555', style='->', connectionstyle="arc3,rad=0"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.5, connectionstyle=connectionstyle))

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # --- Phase 1: Knowledge-Driven Task Construction ---
    ax.text(2.5, 7.5, "Phase 1: Knowledge-Driven\nTask Construction", ha='center', fontsize=14, fontweight='bold', color='#01579B')
    
    # Inputs
    _, _, r_flow, _, _, _, b_flow, _ = draw_box(ax, 0.5, 6, 1.5, 0.8, "Traffic Flow\nData", color='#E3F2FD', icon_path='assets/car.png')
    _, _, r_feat, _, _, _, b_feat, _ = draw_box(ax, 3.0, 6, 1.5, 0.8, "Link\nFeatures", color='#E3F2FD', icon_path='assets/list.png')
    
    # Fusion
    c_fusion_x, c_fusion_y, _, _, _, _, _, b_fusion = draw_box(ax, 1.75, 4.5, 1.5, 0.8, "Feature\nFusion", color='#B3E5FC')
    
    draw_arrow(ax, 1.25, 6, 2.5, 5.3) # Flow to Fusion
    draw_arrow(ax, 3.75, 6, 2.5, 5.3) # Feat to Fusion
    
    # Clustering
    c_kmeans_x, c_kmeans_y, _, _, _, _, _, b_kmeans = draw_box(ax, 1.75, 3.0, 1.5, 0.8, "K-Means\nClustering", color='#81D4FA')
    draw_arrow(ax, c_fusion_x, b_fusion, c_kmeans_x, c_kmeans_y + 0.4)
    
    # Tasks
    _, _, _, _, _, _, _, b_task1 = draw_box(ax, 0.5, 1.0, 1.2, 0.8, "Task 1\n(Highway)", color='#4FC3F7')
    _, _, _, _, _, _, _, b_task2 = draw_box(ax, 1.9, 1.0, 1.2, 0.8, "Task 2\n(Urban)", color='#4FC3F7')
    _, _, _, _, _, _, _, b_taskk = draw_box(ax, 3.3, 1.0, 1.2, 0.8, "Task K\n...", color='#4FC3F7')
    
    draw_arrow(ax, c_kmeans_x, b_kmeans, 1.1, 1.8)
    draw_arrow(ax, c_kmeans_x, b_kmeans, 2.5, 1.8)
    draw_arrow(ax, c_kmeans_x, b_kmeans, 3.9, 1.8)
    
    # Separator
    ax.plot([5, 5], [0.5, 7.5], color='#CCCCCC', linestyle='--', linewidth=2)

    # --- Phase 2: MAML Meta-Training ---
    ax.text(8, 7.5, "Phase 2: MAML\nMeta-Training", ha='center', fontsize=14, fontweight='bold', color='#E65100')
    
    # Global Theta
    c_theta_x, c_theta_y, r_theta, _, l_theta, _, b_theta, _ = draw_box(ax, 7.25, 6, 1.5, 0.8, r"Global $\theta$", color='#FFE0B2', edge_color='#E65100', icon_path='assets/memory.png')
    
    # Inner Loop Box (Container)
    rect = patches.FancyBboxPatch((6, 2.5), 4, 2.5, boxstyle="round,pad=0.1", linewidth=1, edgecolor='#FFCC80', facecolor='#FFF3E0', linestyle='--')
    ax.add_patch(rect)
    ax.text(8, 4.8, "Inner Loop (Fast Adaptation)", ha='center', fontsize=10, color='#E65100')
    
    # Task Specific Theta
    c_local_x, c_local_y, r_local, _, l_local, _, b_local, _ = draw_box(ax, 7.25, 3.0, 1.5, 0.8, r"Task $\theta'_i$", color='#FFCC80', edge_color='#E65100')
    
    # Arrows for Inner Loop
    draw_arrow(ax, c_theta_x, b_theta, c_local_x, c_local_y + 0.4, style='->', connectionstyle="arc3,rad=0")
    ax.text(8.1, 5.4, "Clone", fontsize=9)
    
    # Gradient Update
    ax.annotate("", xy=(8.75, 3.4), xytext=(8.75, 3.4), arrowprops=dict(arrowstyle="->", color="#E65100", lw=1.5, connectionstyle="arc3,rad=2"))
    ax.text(9.2, 3.4, "Gradient\nDescent", fontsize=9)

    # Outer Loop
    draw_arrow(ax, c_local_x, b_local, 10.5, 3.4, style='->') # To Loss
    
    # Loss Calculation
    c_loss_x, c_loss_y, _, _, l_loss, _, _, _ = draw_box(ax, 10.5, 3.0, 1.5, 0.8, "Query Set\nLoss", color='#FFB74D', edge_color='#E65100')
    
    # Backprop
    draw_arrow(ax, c_loss_x, c_loss_y + 0.4, r_theta, 6.4, style='->', connectionstyle="arc3,rad=-0.3")
    ax.text(10.5, 5.5, "Meta-Update", fontsize=10, fontweight='bold', color='#E65100')

    # Separator
    ax.plot([11, 11], [0.5, 7.5], color='#CCCCCC', linestyle='--', linewidth=2)

    # --- Phase 3: RL-based Test-Time Adaptation ---
    ax.text(13.5, 7.5, "Phase 3: RL-based\nTest-Time Adaptation", ha='center', fontsize=14, fontweight='bold', color='#1B5E20')
    
    # RL Agent
    c_agent_x, c_agent_y, _, _, _, _, b_agent, _ = draw_box(ax, 12.75, 6, 1.5, 0.8, "RL Agent", color='#FFF9C4', edge_color='#FBC02D', icon_path='assets/robot.png')
    
    # Environment / Model
    c_model_x, c_model_y, r_model, _, l_model, _, _, t_model = draw_box(ax, 12.75, 3.0, 1.5, 0.8, "LSTM Model\n(Current)", color='#C8E6C9', edge_color='#2E7D32')
    
    # Feedback Loop
    # State/Reward
    draw_arrow(ax, r_model, 3.4, 14.5, 4.5, style='-', connectionstyle="arc3,rad=0")
    draw_arrow(ax, 14.5, 4.5, 14.25, 6.4, style='->', connectionstyle="arc3,rad=0")
    ax.text(14.8, 4.0, "State &\nReward", fontsize=9)
    
    # Actions
    # 1. Keep
    draw_arrow(ax, 12.75, 6, 12.75, 3.8, style='->')
    ax.text(12.85, 5.0, "Action", fontsize=9, fontweight='bold')
    
    # Action Bubbles
    bbox_props = dict(boxstyle="round,pad=0.3", fc="#FFF9C4", ec="#FBC02D", lw=1)
    ax.text(11.8, 4.8, "1. Keep", fontsize=8, bbox=bbox_props)
    ax.text(12.75, 4.8, "2. Fine-tune", fontsize=8, bbox=bbox_props, ha='center')
    ax.text(13.9, 4.8, "3. Reset", fontsize=8, bbox=bbox_props, color='#D32F2F', fontweight='bold')
    
    # Add Reset Icon
    add_image(ax, 'assets/refresh.png', 14.2, 4.8, zoom=0.25)
    
    # Reset Arrow
    draw_arrow(ax, 13.5, 6, 13.8, 3.8, style='->', connectionstyle="arc3,rad=-0.2", color='#D32F2F')

    # Input Data Stream
    ax.text(11.5, 2.5, "New Data", fontsize=10)
    draw_arrow(ax, 11.5, 2.8, l_model, 3.4, style='->')
    
    # Prediction
    draw_arrow(ax, 13.5, 2.2, 13.5, 1.5, style='->')
    ax.text(13.5, 1.2, "Prediction", ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('MetaSTC_Architecture.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('MetaSTC_Architecture.png', dpi=300, bbox_inches='tight')
    print("Diagrams saved as MetaSTC_Architecture.pdf and MetaSTC_Architecture.png")

if __name__ == "__main__":
    create_architecture_diagram()
