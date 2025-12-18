import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import csv
import time
from dataset import HAADFDataset
from models import SigLIPModel
from loss_fn import SigmoidCLIPLoss

def train():

    CONFIG = {
        "h5_path": "/content/2k_normalized.h5",
        "batch_size": 512,    
        "epochs": 200,   
        "lr": 1e-4,           
        "embed_dim": 128,
        "meta_dim": 14,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_dir": "checkpoints_2k",
        "num_workers": 8      
    }
    
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    
    log_file_path = os.path.join(CONFIG["save_dir"], "training_log.csv")
    
    if not os.path.exists(log_file_path):
        with open(log_file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Temperature', 'Learning_Rate', 'Time_Sec'])

    print(f"--- Starting Training on {CONFIG['device']} ---")
    print(f"Logs will be saved to: {log_file_path}")

    print("Initializing Datasets...")
    train_dataset = HAADFDataset(CONFIG["h5_path"], split='train', train_ratio=0.9)
    val_dataset = HAADFDataset(CONFIG["h5_path"], split='val', train_ratio=0.9)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=CONFIG["num_workers"], 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )

    model = SigLIPModel(
        meta_input_dim=CONFIG["meta_dim"], 
        embed_dim=CONFIG["embed_dim"]
    ).to(CONFIG["device"])
    
    criterion = SigmoidCLIPLoss(epsilon=1e-5)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.01)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(CONFIG["epochs"]):
        epoch_start = time.time()
        model.train()
        train_loss_accum = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        current_temp = 0.0
        
        for imgs, metas in pbar:
            imgs = imgs.to(CONFIG["device"])
            metas = metas.to(CONFIG["device"])
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                img_emb, meta_emb, temp, bias = model(imgs, metas)
                loss = criterion(img_emb, meta_emb, temp, bias, raw_meta_batch=metas)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_accum += loss.item()
            current_temp = temp.item()
            pbar.set_postfix(loss=loss.item(), temp=f"{current_temp:.2f}")

        avg_train_loss = train_loss_accum / len(train_loader)
        
        
        model.eval()
        val_loss_accum = 0.0
        
        with torch.no_grad():
            for imgs, metas in val_loader:
                imgs = imgs.to(CONFIG["device"])
                metas = metas.to(CONFIG["device"])
                
                with torch.amp.autocast('cuda'):
                    img_emb, meta_emb, temp, bias = model(imgs, metas)
                    loss = criterion(img_emb, meta_emb, temp, bias, raw_meta_batch=metas)
                    
                val_loss_accum += loss.item()
                
        avg_val_loss = val_loss_accum / len(val_loader)
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        epoch_duration = time.time() - epoch_start
        
        print(f"Summary Ep {epoch+1}: Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | Time {epoch_duration:.1f}s")
        
        with open(log_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                f"{avg_train_loss:.5f}", 
                f"{avg_val_loss:.5f}", 
                f"{current_temp:.4f}",
                f"{current_lr:.2e}",
                f"{epoch_duration:.1f}"
            ])
        
        torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "last.pth"))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best.pth"))
            print(">>> New Best Model Saved!")
            
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], f"epoch_{epoch+1}.pth"))

    total_time = (time.time() - start_time) / 3600
    print(f"Training Complete. Total time: {total_time:.2f} hours.")

if __name__ == "__main__":
    train()