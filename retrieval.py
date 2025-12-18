import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import h5py
import os
import random

from models import SigLIPModel
from dataset import HAADFDataset

# --- CONFIG ---
CHECKPOINT = "checkpoints_2k/best.pth"
H5_PATH = "/content/2k_normalized.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_QUERIES = 2  # Generates 2 examples (4 rows of plots total)

def load_denorm_stats(h5_path):
    """Loads linear min/max stats from H5 file"""
    with h5py.File(h5_path, 'r') as f:
        stats_min = f['stats_min'][:]
        stats_max = f['stats_max'][:]
        if 'metadata_labels' in f:
            labels = [l.decode('utf-8') for l in f['metadata_labels'][:]]
        else:
            labels = [f"Param_{i}" for i in range(len(stats_min))]

    stats_range = stats_max - stats_min
    return stats_min, stats_range, labels

def inverse_transform_linear(norm_vec, stats_min, stats_range):
    """Standard Linear De-normalization: Phys = Norm * Range + Min"""
    phys = norm_vec * stats_range + stats_min
    return phys

def create_meta_string(phys_vec, labels):
    """Formats all parameters into a clean multi-line string"""
    lines = []
    for i, (label, val) in enumerate(zip(labels, phys_vec)):
        # Smart formatting: Scientific notation if very small or very large
        if abs(val) < 1e-3 and val != 0:
            val_str = f"{val:.2e}"
        elif abs(val) > 1000:
            val_str = f"{val:.1e}"
        else:
            val_str = f"{val:.4f}"
            
        # Truncate label to fit in plot
        clean_label = label.split('(')[0].strip()[:18] 
        lines.append(f"{clean_label:<18}: {val_str}")
    return "\n".join(lines)

def run_full_retrieval():
    print(f"--- Running Full Metadata Retrieval Demo (Linear) ---")
    
    # 1. Load Model
    # Note: Ensure meta_dim matches your training (13 or 14)
    model = SigLIPModel(meta_input_dim=14, embed_dim=128).to(DEVICE)
    
    if os.path.exists(CHECKPOINT):
        model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    else:
        print(f"Checkpoint not found at {CHECKPOINT}")
        return
    model.eval()
    
    # 2. Setup Data
    stats_min, stats_range, labels = load_denorm_stats(H5_PATH)
    
    print("Loading Validation Dataset...")
    val_dataset = HAADFDataset(H5_PATH, split='val')
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    # 3. Index the Gallery
    print("Indexing Validation Gallery...")
    db_img_embs = []
    
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(DEVICE)
            emb = model.visual(imgs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            db_img_embs.append(emb)
    
    db_img_embs = torch.cat(db_img_embs)

    # 4. Pick Random Queries
    # We pick random indices every time script is run
    query_indices = random.sample(range(len(val_dataset)), NUM_QUERIES)

    # 5. Setup Grid
    # Rows = NUM_QUERIES * 2 (Row for Images, Row for Text)
    # Cols = 4 (1 Query + 3 Matches)
    total_rows = NUM_QUERIES * 2
    # Tall figure to accommodate text rows
    fig, axes = plt.subplots(total_rows, 4, figsize=(16, 8 * total_rows))
    plt.subplots_adjust(hspace=0.2, wspace=0.1)

    for q_num, q_idx in enumerate(query_indices):
        row_img = q_num * 2
        row_txt = q_num * 2 + 1
        
        # --- A. Prepare Query Info ---
        q_img, q_meta_norm = val_dataset[q_idx]
        q_phys = inverse_transform_linear(q_meta_norm.numpy(), stats_min, stats_range)
        
        # Encode Query Metadata
        q_vec = q_meta_norm.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_emb = model.meta(q_vec)
            q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)
            
        # Search
        sims = q_emb @ db_img_embs.t()
        
        # Get Top Matches
        scores, top_indices = sims.topk(10, dim=1)
        top_indices = top_indices[0].cpu().numpy()
        scores = scores[0].cpu().numpy()
        
        # Filter: Exclude the query itself from results to show true retrieval
        matches = []
        match_scores = []
        for idx, score in zip(top_indices, scores):
            if idx != q_idx:
                matches.append(idx)
                match_scores.append(score)
            if len(matches) == 3:
                break
        
        # --- B. Plot Column 0 (The Query / Target) ---
        
        # Image
        ax_q_img = axes[row_img, 0]
        ax_q_img.imshow(q_img[0].cpu().numpy(), cmap='gray')
        ax_q_img.set_title(f"QUERY TARGET\n(Index {q_idx})", fontweight='bold', color='black', pad=10)
        ax_q_img.axis('off')
        
        # Metadata Text
        ax_q_txt = axes[row_txt, 0]
        meta_str = create_meta_string(q_phys, labels)
        ax_q_txt.text(0.05, 0.95, meta_str, transform=ax_q_txt.transAxes, 
                      fontsize=10, va='top', family='monospace')
        ax_q_txt.axis('off')
        
        # --- C. Plot Columns 1-3 (The Retrievals) ---
        for k, (r_idx, score) in enumerate(zip(matches, match_scores)):
            col = k + 1
            
            # Fetch Match Data
            r_img, r_meta_norm = val_dataset[r_idx]
            r_phys = inverse_transform_linear(r_meta_norm.numpy(), stats_min, stats_range)
            
            # Image
            ax_img = axes[row_img, col]
            ax_img.imshow(r_img[0].cpu().numpy(), cmap='gray')
            ax_img.set_title(f"Match #{k+1}\nSim: {score:.3f}", color='blue')
            ax_img.axis('off')
            
            # Metadata Text
            ax_txt = axes[row_txt, col]
            match_meta_str = create_meta_string(r_phys, labels)
            ax_txt.text(0.05, 0.95, match_meta_str, transform=ax_txt.transAxes, 
                        fontsize=9, va='top', family='monospace', color='#333333')
            ax_txt.axis('off')

    print("Displaying Results...")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_full_retrieval()