import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py
import os

from models import SigLIPModel
from dataset import HAADFDataset

# --- CONFIG ---
CHECKPOINT = "checkpoints_2k/best.pth"
H5_PATH = "/content/2k_normalized.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512
K_NEIGHBORS = 5

def load_denorm_stats(h5_path):
    with h5py.File(h5_path, 'r') as f:
        stats_min = torch.from_numpy(f['stats_min'][:]).to(DEVICE)
        stats_max = torch.from_numpy(f['stats_max'][:]).to(DEVICE)
        
        if 'metadata_labels' in f:
            labels = [l.decode('utf-8') for l in f['metadata_labels'][:]]
        else:
            labels = [f"Param_{i}" for i in range(len(stats_min))]
    return stats_min, stats_max, labels

def get_valid_levels(memory_metas, max_unique=10):
    """
    Analyzes the Training Data to find valid discrete levels for each parameter.
    Returns a list where each element is either:
      - A torch Tensor of valid values (if discrete)
      - None (if continuous)
    """
    valid_levels = []
    print("Analyzing parameter discreteness...")
    
    # Iterate over columns (parameters)
    for i in range(memory_metas.shape[1]):
        col = memory_metas[:, i]
        unique_vals = torch.unique(col)
        
        # If there are few unique values, we treat this parameter as Discrete
        if len(unique_vals) <= max_unique:
            valid_levels.append(unique_vals)
        else:
            valid_levels.append(None) # Continuous
            
    return valid_levels

def snap_to_levels(predictions, valid_levels):
    """
    Snaps continuous predictions to the nearest valid level found in training data.
    """
    snapped_preds = predictions.clone()
    
    for i, levels in enumerate(valid_levels):
        if levels is not None:
            # We have a discrete parameter. Snap values.
            # Shape: (N, 1) vs (1, L) -> Broadcast
            # Find index of minimal distance
            col_preds = predictions[:, i].unsqueeze(1) # (N, 1)
            levels = levels.unsqueeze(0)               # (1, L)
            
            dists = torch.abs(col_preds - levels)      # (N, L)
            min_indices = torch.argmin(dists, dim=1)   # (N,)
            
            # Map back to values
            snapped_preds[:, i] = levels[0, min_indices]
            
    return snapped_preds

def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0: return 0.0
    return 1 - (ss_res / ss_tot)

def build_memory_bank(model, loader):
    print("Building Knowledge Base from Training Data...")
    mem_feats = []
    mem_metas = []
    with torch.no_grad():
        for imgs, metas in tqdm(loader):
            imgs, metas = imgs.to(DEVICE), metas.to(DEVICE)
            feat = model.visual(imgs)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            mem_feats.append(feat)
            mem_metas.append(metas)
    return torch.cat(mem_feats), torch.cat(mem_metas)

def run_prediction_demo():
    print(f"--- Running Physics Prediction (With Discrete Snapping) ---")
    
    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found at {CHECKPOINT}")
        return

    model = SigLIPModel(meta_input_dim=14, embed_dim=128).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    
    stats_min, stats_max, labels = load_denorm_stats(H5_PATH)
    stats_range = stats_max - stats_min
    
    train_loader = DataLoader(HAADFDataset(H5_PATH, split='train'), batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(HAADFDataset(H5_PATH, split='val'), batch_size=BATCH_SIZE, shuffle=False)

    mem_feats, mem_metas_norm = build_memory_bank(model, train_loader)
    
    #Analyze Training Data for Discrete Levels
    mem_metas_phys = mem_metas_norm * stats_range + stats_min
    valid_levels = get_valid_levels(mem_metas_phys)

    print("\nPredicting on Validation Set...")
    preds_phys = []
    truths_phys = []
    
    with torch.no_grad():
        for i, (imgs, metas) in enumerate(val_loader):
            
            imgs, metas = imgs.to(DEVICE), metas.to(DEVICE)
            
            query = model.visual(imgs)
            query = query / query.norm(dim=-1, keepdim=True)
            
            # Neighbors
            sims = query @ mem_feats.t()
            scores, indices = sims.topk(K_NEIGHBORS, dim=1)
            
            weights = torch.softmax(scores * 10, dim=1).unsqueeze(-1)
            neighbor_vals = mem_metas_norm[indices]
            pred_norm = (neighbor_vals * weights).sum(dim=1)
            
            pred_p = pred_norm * stats_range + stats_min
            true_p = metas * stats_range + stats_min
            
            preds_phys.append(pred_p)
            truths_phys.append(true_p)
            
    preds_phys = torch.cat(preds_phys)
    truths_phys = torch.cat(truths_phys)

    # 6. Apply Snapping (The "Nearest Valid Value" logic)
    print("Snapping predictions to nearest valid physical levels...")
    preds_phys_snapped = snap_to_levels(preds_phys, valid_levels)

    # Convert to numpy for plotting
    y_preds = preds_phys_snapped.cpu().numpy()
    y_trues = truths_phys.cpu().numpy()
    
    # 7. Visualization
    plot_indices = list(range(14))
    fig, axes = plt.subplots(7, 2, figsize=(10, 35))
    axes = axes.flatten()
    
    print("\n---------------- METRICS (Sorted by R2) ----------------")
    print(f"{'PARAMETER':<30} | {'MAPE %':<10} | {'ACC %':<10} | {'R2':<6} | {'TYPE'}")
    print("-" * 80)
    
    metrics = []

    for i in plot_indices:
        label = labels[i]
        yp = y_preds[:, i]
        yt = y_trues[:, i]
        is_discrete = valid_levels[i] is not None
        
        # --- Metric: MAPE (Mean Absolute Percentage Error) ---
        # Handle zeros to avoid infinity
        mask = np.abs(yt) > 1e-9
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((yp[mask] - yt[mask]) / yt[mask])) * 100
        else:
            mape = 0.0
            
        # --- Metric: Exact Accuracy (For discrete) ---
        # We consider it "Accurate" if it matches within a tiny tolerance
        if is_discrete:
            matches = np.abs(yp - yt) < 1e-6
            acc = np.mean(matches) * 100
        else:
            acc = np.nan # Not applicable for continuous

        # --- Metric: R2 Score ---
        r2 = calculate_r2(yt, yp)

        metrics.append({
            'idx': i,
            'label': label,
            'mape': mape,
            'acc': acc,
            'r2': r2,
            'type': "Discrete" if is_discrete else "Cont.",
            'yp': yp,
            'yt': yt
        })
        
        # Console Output
        acc_str = f"{acc:.1f}" if is_discrete else "N/A"
        print(f"{label:<30} | {mape:.2f}%     | {acc_str:<10} | {r2:.3f}  | {'Discrete' if is_discrete else 'Continuous'}")

    # Plotting Loop
    for i, ax in enumerate(axes):
        if i >= len(metrics):
            ax.axis('off')
            continue
            
        m = metrics[i]
        
        # Scatter
        ax.scatter(m['yt'], m['yp'], alpha=0.4, s=15, c='#1f77b4', edgecolors='none')
        
        # Limits
        all_vals = np.concatenate([m['yt'], m['yp']])
        val_min, val_max = np.min(all_vals), np.max(all_vals)
        pad = (val_max - val_min) * 0.05 if (val_max - val_min) > 0 else 0.1
        
        ax.plot([val_min-pad, val_max+pad], [val_min-pad, val_max+pad], 'r--', alpha=0.8, label='Perfect')
        
        # Enforce Square
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(val_min-pad, val_max+pad)
        ax.set_ylim(val_min-pad, val_max+pad)
        
        # Title logic
        if m['type'] == "Discrete":
            title_str = f"{m['label']}\nAcc: {m['acc']:.1f}% | MAPE: {m['mape']:.1f}%"
        else:
            title_str = f"{m['label']}\nR²: {m['r2']:.2f} | MAPE: {m['mape']:.1f}%"
            
        ax.set_title(title_str, fontsize=10, fontweight='bold')
        if i % 2 == 0: ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_prediction_demo()