import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx
import os

# Set global font style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Bitstream Vera Sans']

# Color Palette (Publication Ready - Soft & Professional)
COLORS = {
    'text': '#2C3E50',
    'arrow': '#546E7A',
    'box_edge': '#455A64',
    
    # Section Backgrounds
    'bg_input': '#E3F2FD',      # Light Blue
    'bg_middle': '#F3E5F5',     # Light Purple
    'bg_right': '#FFF3E0',      # Light Orange
    
    # Section Borders
    'border_input': '#64B5F6',
    'border_middle': '#BA68C8',
    'border_right': '#FFB74D',
    
    # Element Colors
    'node_color': '#1E88E5',
    'plot_line': '#3949AB',
    'box_spatial': '#FFFFFF',
    'box_temporal': '#FFFFFF',
    'box_meta': '#FFE0B2',      # Orange tint
    'box_grad': '#FFCCBC',      # Red tint
    'box_update': '#FFE0B2',
    'box_memory': '#9FA8DA',    # Indigo tint
    
    # Accents
    'cluster_1': '#FDD835', # Yellow
    'cluster_2': '#8E24AA', # Purple
    'cluster_3': '#00897B', # Teal
    'cluster_4': '#1E88E5', # Blue
    'cluster_5': '#43A047', # Green
}

def draw_shadow_box(ax, x, y, w, h, text, facecolor='#FFFFFF', edgecolor=COLORS['box_edge'], 
                   fontsize=10, boxstyle='round,pad=0.02', zorder=10, text_color=COLORS['text'], fontweight='normal'):
    # Shadow (offset)
    shadow = patches.FancyBboxPatch((x+0.06, y-0.06), w, h, boxstyle=boxstyle, 
                                   linewidth=0, facecolor='#000000', alpha=0.15, zorder=zorder-1)
    ax.add_patch(shadow)
    
    # Main Box
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle=boxstyle, 
                                 linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=zorder)
    ax.add_patch(box)
    
    # Text
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, 
            color=text_color, fontweight=fontweight, zorder=zorder+1)
    
    return {'x': x + w/2, 'y': y + h/2, 'top': y+h, 'bottom': y, 'left': x, 'right': x+w}

def draw_fancy_arrow(ax, x1, y1, x2, y2, color=COLORS['arrow'], style='->', connectionstyle='arc3,rad=0', lw=1.5, ls='-'):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, connectionstyle=connectionstyle, linestyle=ls),
                zorder=5)

def create_framework_diagram():
    fig, ax = plt.subplots(figsize=(24, 10))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    # ==================== 1. Input Section (Left) ====================
    # Background Area
    rect_input = patches.FancyBboxPatch((0.5, 0.5), 4.5, 10.0, boxstyle="round,pad=0.2", 
                                      linewidth=2, edgecolor=COLORS['border_input'], facecolor=COLORS['bg_input'], 
                                      linestyle='--', alpha=0.3, zorder=0)
    ax.add_patch(rect_input)
    ax.text(2.75, 10.2, "Input Data", ha='center', fontsize=16, fontweight='bold', color=COLORS['text'])

    # Road Network Graph
    G = nx.erdos_renyi_graph(12, 0.35, seed=42)
    pos = nx.spring_layout(G, center=(2.75, 7.8), scale=1.5)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#546E7A', width=1.5, alpha=0.7)
    # Draw nodes (with white border for pop)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=200, node_color=COLORS['node_color'], 
                           edgecolors='white', linewidths=1.5)
    
    ax.text(2.75, 5.8, "Road Network G(V, E)", ha='center', fontsize=13, fontweight='bold', color=COLORS['text'])

    # Traffic Flow Plots
    # Create mini-axes for plots
    for i in range(4):
        # Position: x, y, w, h
        ax_ins = ax.inset_axes([0.8 + (i%2)*1.9, 1.5 + (1-i//2)*1.6, 1.6, 1.1], transform=ax.transData)
        x_data = np.linspace(0, 10, 60)
        y_data = np.sin(x_data + i*1.5) + np.random.normal(0, 0.15, 60) + 2
        
        ax_ins.plot(x_data, y_data, color=COLORS['plot_line'], lw=1.5)
        ax_ins.set_xticks([])
        ax_ins.set_yticks([])
        ax_ins.set_title(f"Link {i+1}", fontsize=9, color=COLORS['text'])
        ax_ins.patch.set_alpha(0.6) # Transparent background
        ax_ins.grid(True, linestyle=':', alpha=0.5)
        
        # Add a nice border
        for spine in ax_ins.spines.values():
            spine.set_edgecolor('#B0BEC5')
            spine.set_linewidth(1)
    
    ax.text(2.75, 0.8, "Traffic Flow Sequences Xt", ha='center', fontsize=13, fontweight='bold', color=COLORS['text'])

    # ==================== 2. Spatio-Temporal Representation Clustering (Middle) ====================
    # Background Area
    rect_mid = patches.FancyBboxPatch((5.8, 0.5), 10.5, 10.0, boxstyle="round,pad=0.2", 
                                    linewidth=2, edgecolor=COLORS['border_middle'], facecolor=COLORS['bg_middle'], 
                                    linestyle='--', alpha=0.3, zorder=0)
    ax.add_patch(rect_mid)
    ax.text(11.05, 10.2, "Spatio-Temporal Representation Clustering", ha='center', fontsize=16, fontweight='bold', color=COLORS['text'])

    # -- Spatial Branch --
    ax.text(8.0, 9.4, "Static Spatial Features", ha='center', fontsize=12, color=COLORS['text'])
    
    # Feature boxes
    feat_w, feat_h = 1.4, 0.6
    f1 = draw_shadow_box(ax, 6.2, 8.6, feat_w, feat_h, "Road ID", facecolor='white', edgecolor=COLORS['border_middle'], boxstyle='round,pad=0.02')
    f2 = draw_shadow_box(ax, 7.8, 8.6, feat_w, feat_h, "Level", facecolor='white', edgecolor=COLORS['border_middle'], boxstyle='round,pad=0.02')
    f3 = draw_shadow_box(ax, 9.4, 8.6, feat_w, feat_h, "Lanes/Geo", facecolor='white', edgecolor=COLORS['border_middle'], boxstyle='round,pad=0.02')

    # Embed boxes
    e1 = draw_shadow_box(ax, 6.2, 7.4, feat_w, feat_h, "Embed", facecolor='#F3E5F5')
    e2 = draw_shadow_box(ax, 7.8, 7.4, feat_w, feat_h, "Embed", facecolor='#F3E5F5')
    e3 = draw_shadow_box(ax, 9.4, 7.4, feat_w, feat_h, "Embed", facecolor='#F3E5F5')

    # Arrows Feat -> Embed
    for f, e in zip([f1, f2, f3], [e1, e2, e3]):
        draw_fancy_arrow(ax, f['x'], f['bottom'], e['x'], e['top'], color=COLORS['border_middle'])

    # Spatial Encoding
    s_enc = draw_shadow_box(ax, 7.0, 6.2, 3.0, 0.7, "Spatial Encoding", facecolor='#E1BEE7', fontweight='bold')
    
    # Arrows Embed -> S_Enc
    draw_fancy_arrow(ax, e1['x'], e1['bottom'], s_enc['x']-1.0, s_enc['top'], color=COLORS['border_middle'])
    draw_fancy_arrow(ax, e2['x'], e2['bottom'], s_enc['x'], s_enc['top'], color=COLORS['border_middle'])
    draw_fancy_arrow(ax, e3['x'], e3['bottom'], s_enc['x']+1.0, s_enc['top'], color=COLORS['border_middle'])

    # -- Temporal Branch --
    t2v = draw_shadow_box(ax, 7.0, 1.5, 3.0, 0.7, "Traffic2Vec\n(Linear+Sin)", facecolor='#E1BEE7', fontsize=9)
    t_enc = draw_shadow_box(ax, 7.0, 2.7, 3.0, 0.7, "Dynamic Temporal", facecolor='#E1BEE7', fontweight='bold')
    draw_fancy_arrow(ax, t2v['x'], t2v['top'], t_enc['x'], t_enc['bottom'], color=COLORS['border_middle'])

    # -- Gating --
    gating = draw_shadow_box(ax, 7.5, 4.5, 2.0, 0.9, "S-T Gating\nUnit", facecolor='#CE93D8', fontweight='bold', fontsize=11)
    ax.text(9.8, 4.9, "$W_{st}, W_{ts}$", fontsize=10, color=COLORS['text']) # Weights annotation
    
    draw_fancy_arrow(ax, s_enc['x'], s_enc['bottom'], gating['x'], gating['top'], color=COLORS['border_middle'])
    draw_fancy_arrow(ax, t_enc['x'], t_enc['top'], gating['x'], gating['bottom'], color=COLORS['border_middle'])

    # -- Clustering Visualization --
    # Scatter plot
    ax_scatter = ax.inset_axes([11.5, 3.5, 4.0, 4.0], transform=ax.transData)
    # Generate clusters
    np.random.seed(10)
    c1_x = np.random.normal(0.25, 0.08, 40)
    c1_y = np.random.normal(0.25, 0.08, 40)
    c2_x = np.random.normal(0.75, 0.08, 40)
    c2_y = np.random.normal(0.25, 0.08, 40)
    c3_x = np.random.normal(0.5, 0.08, 40)
    c3_y = np.random.normal(0.75, 0.08, 40)
    
    ax_scatter.scatter(c1_x, c1_y, c=COLORS['cluster_1'], s=15, alpha=0.8, edgecolors='none')
    ax_scatter.scatter(c2_x, c2_y, c=COLORS['cluster_2'], s=15, alpha=0.8, edgecolors='none')
    ax_scatter.scatter(c3_x, c3_y, c=COLORS['cluster_3'], s=15, alpha=0.8, edgecolors='none')
    
    # Centroids
    ax_scatter.scatter([0.25, 0.75, 0.5], [0.25, 0.25, 0.75], c='black', s=60, marker='X', edgecolors='white', linewidth=1)
    ax_scatter.text(0.25, 0.35, "$p_1$", fontsize=12, fontweight='bold')
    ax_scatter.text(0.75, 0.35, "$p_K$", fontsize=12, fontweight='bold')
    ax_scatter.text(0.5, 0.65, "$p_2$", fontsize=12, fontweight='bold')
    
    ax_scatter.axis('off')
    ax.text(13.5, 6.5, "K-Means++", fontsize=13, fontweight='bold', color=COLORS['text'])
    ax.text(13.5, 7.6, "Clustering", fontsize=12, color=COLORS['text'])
    draw_fancy_arrow(ax, 13.5, 7.4, 13.5, 6.8, color=COLORS['border_middle']) # Arrow down to clusters

    # Connection Gating -> Clustering
    draw_fancy_arrow(ax, gating['right'], gating['y'], 11.5, 5.5, color=COLORS['border_middle']) # To scatter plot area

    # Representation h (visualized as bars)
    ax.text(13.5, 9.0, "Spatio-Temporal Representation $h$", ha='center', fontsize=12, color=COLORS['text'])
    for i in range(5):
        rect_h = patches.FancyBboxPatch((11.8 + i*0.7, 8.2), 0.5, 0.4, boxstyle="round,pad=0.05", 
                                      facecolor='#E0E0E0', edgecolor='#757575')
        ax.add_patch(rect_h)
    
    # Connection h -> Clustering
    draw_fancy_arrow(ax, 13.5, 8.2, 13.5, 7.8, color=COLORS['border_middle'])

    # Clustering Representation (bottom)
    ax.text(13.5, 2.8, "Cluster Centroids $p_k$", ha='center', fontsize=12, color=COLORS['text'])
    c_colors = [COLORS['cluster_1'], COLORS['cluster_2'], COLORS['cluster_3'], COLORS['cluster_4'], COLORS['cluster_5']]
    for i in range(5):
        rect_c = patches.FancyBboxPatch((11.8 + i*0.7, 2.0), 0.5, 0.4, boxstyle="round,pad=0.05",
                                      facecolor=c_colors[i], edgecolor='#757575')
        ax.add_patch(rect_c)
    
    draw_fancy_arrow(ax, 13.5, 3.5, 13.5, 2.5, color=COLORS['border_middle']) # Scatter -> Rep

    # ==================== 3. Spatio-Temporal Meta Learner (Right) ====================
    # Background Area
    rect_right = patches.FancyBboxPatch((16.8, 0.5), 6.7, 10.0, boxstyle="round,pad=0.2", 
                                      linewidth=2, edgecolor=COLORS['border_right'], facecolor=COLORS['bg_right'], 
                                      linestyle='--', alpha=0.3, zorder=0)
    ax.add_patch(rect_right)
    ax.text(20.15, 10.2, "Spatio-Temporal Meta Learner", ha='center', fontsize=16, fontweight='bold', color=COLORS['text'])

    # --- Cluster-aware Contextual Memory Unit ---
    # Container Box
    mem_container = patches.FancyBboxPatch((17.2, 7.5), 5.9, 2.2, boxstyle="round,pad=0.1",
                                         linewidth=1, edgecolor='#9FA8DA', facecolor='white', alpha=0.8)
    ax.add_patch(mem_container)
    ax.text(20.15, 9.4, "Cluster-aware Contextual Memory Unit", ha='center', fontsize=11, fontweight='bold', color='#3F51B5')

    # Similarity & Softmax
    sim_box = draw_shadow_box(ax, 17.4, 8.2, 1.2, 0.8, "Sim\n$a_j^k$", facecolor='#E8EAF6', fontsize=9)
    softmax_box = draw_shadow_box(ax, 18.8, 8.2, 1.2, 0.8, "Softmax", facecolor='#E8EAF6', fontsize=9)
    
    # Memory Matrix M_c
    mem_matrix = draw_shadow_box(ax, 20.3, 8.0, 1.2, 1.0, "Memory\n$M_c$", facecolor=COLORS['box_memory'], fontweight='bold')
    
    # Gated Update
    gate_upd = draw_shadow_box(ax, 21.8, 8.2, 1.0, 0.8, "Gated\n$r_t, u_t$", facecolor='#C5CAE9', fontsize=9)

    # Connections inside Memory Unit
    draw_fancy_arrow(ax, sim_box['right'], sim_box['y'], softmax_box['left'], softmax_box['y'], color='#7986CB')
    draw_fancy_arrow(ax, softmax_box['right'], softmax_box['y'], mem_matrix['left'], mem_matrix['y'], color='#7986CB')
    draw_fancy_arrow(ax, gate_upd['left'], gate_upd['y'], mem_matrix['right'], mem_matrix['y'], color='#7986CB', style='->') # Feedback

    # Generated Params
    p_theta = draw_shadow_box(ax, 20.3, 6.8, 1.2, 0.6, r"$\theta_{init}$", facecolor='#FFF9C4', fontweight='bold')
    draw_fancy_arrow(ax, mem_matrix['x'], mem_matrix['bottom'], p_theta['x'], p_theta['top'], color='#7986CB')

    # --- Meta Learning Loop ---
    y_start = 5.5
    y_step = 2.0
    
    for i in range(3):
        y_pos = y_start - i * y_step
        
        # Backbone Model Box
        ml = draw_shadow_box(ax, 17.2, y_pos, 2.0, 1.0, f"Backbone\n(LSTM/FiLM)", 
                            facecolor=COLORS['box_meta'], fontweight='bold', fontsize=10)
        
        # Compute Gradient
        cg = draw_shadow_box(ax, 19.8, y_pos, 1.8, 1.0, "Compute\nGradient", facecolor=COLORS['box_grad'])
        
        # Update
        up = draw_shadow_box(ax, 22.1, y_pos, 1.0, 1.0, "Update", facecolor=COLORS['box_update'])
        
        # Arrows horizontal
        draw_fancy_arrow(ax, ml['right'], ml['y'], cg['left'], cg['y'], color=COLORS['border_right'])
        draw_fancy_arrow(ax, cg['right'], cg['y'], up['left'], up['y'], color=COLORS['border_right'])
        
        # Vertical connections
        if i == 0:
            # Init to first layer
            draw_fancy_arrow(ax, p_theta['x'], p_theta['bottom'], 18.2, y_pos+1.0, color=COLORS['border_right']) # To Backbone
            
            # Annotations for Support/Query
            ax.text(19.8, y_pos + 1.2, "Support Set (Local)", ha='center', fontsize=9, color='#D84315')
            ax.text(22.6, y_pos + 1.2, "Query Set (Global)", ha='center', fontsize=9, color='#EF6C00')

        if i < 2:
            # Layer to Layer
            draw_fancy_arrow(ax, up['x'], up['bottom'], 22.6, y_pos - 1.0, color=COLORS['border_right']) # Update flow down
            
            # Dots
            if i == 1:
                ax.text(18.2, y_pos - 1.0, ":", fontsize=16, fontweight='bold', color=COLORS['text'])
                ax.text(20.7, y_pos - 1.0, ":", fontsize=16, fontweight='bold', color=COLORS['text'])

    # ==================== Global Connections ====================
    # Input -> Middle
    # Graph to Spatial
    draw_fancy_arrow(ax, 4.5, 7.8, 6.2, 8.9, color=COLORS['border_input'], connectionstyle="arc3,rad=-0.1", lw=2) 
    # Flow to Traffic2Vec
    draw_fancy_arrow(ax, 4.5, 2.0, 7.0, 1.8, color=COLORS['border_input'], connectionstyle="arc3,rad=0.1", lw=2)

    # Middle -> Right
    # 1. Representation h -> Similarity
    draw_fancy_arrow(ax, 15.5, 8.4, 17.4, 8.4, color='#757575', lw=2, ls='--')
    
    # 2. Cluster Centroids -> Similarity (Long path)
    draw_fancy_arrow(ax, 15.5, 2.2, 16.5, 2.2, color='#757575', lw=2, ls='--') # Out from clusters
    # Draw a connecting line manually to avoid ugly curves
    ax.plot([16.5, 16.5], [2.2, 8.0], color='#757575', lw=2, ls='--', zorder=1)
    draw_fancy_arrow(ax, 16.5, 8.0, 17.4, 8.2, color='#757575', lw=2, ls='--') # Into Sim box

    # plt.tight_layout() # Disabled to prevent errors with manual layout
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path_png = os.path.join(script_dir, 'framework_beautified.png')
    save_path_pdf = os.path.join(script_dir, 'framework_beautified.pdf')
    
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
    print(f"Framework diagram saved to {script_dir}")

if __name__ == "__main__":
    create_framework_diagram()
