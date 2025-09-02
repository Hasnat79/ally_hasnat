"""
@author: Hasnat Md Abdullah
@date: Aug 28, 2025
"""
import os
import torch
import time 
from tqdm import tqdm
import torch.optim as optim
from model import JEPA_Encoder, JEPA_Predictor, JEPA_FeatureMask
def JEPA_train(dataloader, device, epochs, weight_dir, 
              modelname, proj_dim=256, lr=1e-4, imgformat='rgb',
              momentum=0.99,pretrained=True):
    """
    Train JEPA model.
    Args:
        dataloader: DataLoader yeilding input images
        device: torch.device
        epochs: number of training epochs
        weight_dir: directory to save best weights
        modelname: filename for saving encoder weights
        proj_dim: projection dimension
        lr: learning rate
    """
    context_enc = JEPA_Encoder(proj_dim=proj_dim, imgformat='rgb', pretrained=pretrained).to(device)
    target_enc = JEPA_Encoder(proj_dim=proj_dim, imgformat='rgb', pretrained=pretrained).to(device)

    # Initialize target encoder as copy of context
    target_enc.load_state_dict(context_enc.state_dict())
    for p in target_enc.parameters():
        p.requires_grad = False  # frozen, updated by EMA (Exponential moving average)
    
    predictor = JEPA_Predictor(channels=proj_dim).to(device)
    
    # Combine into JEPA model
    jepa_model = JEPA_FeatureMask(context_enc, target_enc, predictor).to(device)

    # Optimizer
    optimizer = optim.Adam(list(context_enc.parameters()) + list(predictor.parameters()), lr=lr)

    # Tracking
    best_loss = float('inf')


    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(epochs):
        start_time = time.time()
        jepa_model.train()
        running_loss = 0.0
        
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        for batch_idx, imgs in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}")):
            # Move to device
            x = imgs.to(device, non_blocking=True)

            # Forward pass through JEPA
            loss, _, _, _, _ = jepa_model(x)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # EMA update for target encoder
            jepa_model.update_target_encoder(momentum=momentum)

            running_loss += loss.item()
            # Optional: print batch-level info
            print(f"\rBatch [{batch_idx+1}/{len(dataloader)}] --> Batch Loss: {loss.item():.6f}", end='')
        
        # Epoch stats
        avg_loss = running_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch+1} finished. Avg Loss: {avg_loss:.6f} | Time: {elapsed:.2f}s")

        # Save best encoder
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(context_enc.state_dict(), os.path.join(weight_dir, modelname))
            print(f"Model improved. Encoder saved with Avg Loss: {avg_loss:.6f}")
            
    print("\nTraining completed.")
    print(f"Best Loss Achieved: {best_loss:.6f}")
