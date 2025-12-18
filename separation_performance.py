import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import SigLIPModel
from dataset import HAADFDataset

#set torch random seed
torch.manual_seed(67)

# CONFIG
H5_PATH = "/content/2k_normalized.h5"
CHECKPOINT = "checkpoints_2k/best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512

def diagnose():
    print(f"--- Diagnosing Model Health (Full Validation Set) ---")
    
    model = SigLIPModel(meta_input_dim=14, embed_dim=128).to(DEVICE)
    try:
        model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
        print(f"Checkpoint loaded from {CHECKPOINT}")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        return

    model.eval()
    # 2. Load Data
    dataset = HAADFDataset(H5_PATH, split='val')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    all_same_physics_sims = []
    all_diff_physics_sims = []
    
    print(f"Processing {len(dataset)} validation samples...")
    
    with torch.no_grad():
        for imgs, metas in tqdm(loader, desc="Calculating Similarities"):
            imgs, metas = imgs.to(DEVICE), metas.to(DEVICE)
            
            img_emb = model.visual(imgs)
            meta_emb = model.meta(metas)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            meta_emb = meta_emb / meta_emb.norm(dim=-1, keepdim=True)
            
            sim_matrix = img_emb @ meta_emb.t()

            #gt same settings
            meta_dists = torch.cdist(metas, metas, p=2)
            is_same_physics = meta_dists < 1e-4

            same_phys_scores = sim_matrix[is_same_physics].cpu().numpy()
            all_same_physics_sims.append(same_phys_scores)

            diff_phys_scores = sim_matrix[~is_same_physics].cpu().numpy()
            all_diff_physics_sims.append(diff_phys_scores)

    all_same_physics_sims = np.concatenate(all_same_physics_sims)
    all_diff_physics_sims = np.concatenate(all_diff_physics_sims)

    avg_match = np.mean(all_same_physics_sims)
    avg_noise = np.mean(all_diff_physics_sims)
    gap = avg_match - avg_noise
    
    print("\n------------------------------------------------")
    print(f"RESULTS (Corrected Logic):")
    print(f"Avg Similarity (Same Physics):      {avg_match:.4f}")
    print(f"Avg Similarity (Different Physics): {avg_noise:.4f}")
    print(f"Signal-to-Noise Gap:                {gap:.4f}")
    print("------------------------------------------------")

    # 4. Plot
    plt.figure(figsize=(10, 6))
    
    # Plot Noise (Gray)
    plt.hist(all_diff_physics_sims, bins=100, alpha=0.5, label='Image-incorrect Metadata similarity', 
             color='gray', density=True, range=(-1, 1))
    
    # Plot Signal (Green)
    plt.hist(all_same_physics_sims, bins=100, alpha=0.7, label='Image-correct Metadata similarity', 
             color='green', density=True, range=(-1, 1))
    
    plt.axvline(avg_noise, color='gray', linestyle='dashed', linewidth=1)
    plt.axvline(avg_match, color='green', linestyle='dashed', linewidth=1)
    
    plt.title(f"Similarity Distribution \nGap: {gap:.3f}")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    diagnose()