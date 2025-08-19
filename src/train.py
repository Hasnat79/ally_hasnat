"""
Created on Wed May 27 08:51:41 2025

@author: Tiago
"""
import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from src.model import JEPA_Encoder, JEPA_Predictor, JEPA_FeatureMask
from src.model import JEPAUNet
from src.utils import calculate_dice

def JEPAtrain(dataloader, 
              device, 
              epochs, 
              weight_dir, 
              modelname, 
              proj_dim=256, 
              lr=1e-4,
              imgformat='rgb',
              momentum=0.99,
              pretrained=True):
    """
    Train a JEPA model.
    
    Args:
        dataloader: DataLoader yielding input images
        device: torch.device
        epochs: number of epochs
        weight_dir: directory to save best weights
        modelname: filename for saving encoder weights
        proj_dim: projection dimension
        lr: learning rate
    """

    # ----------------------------
    # Initialize encoders and predictor
    # ----------------------------
    context_enc = JEPA_Encoder(proj_dim=proj_dim, imgformat='rgb', pretrained=pretrained).to(device)
    target_enc = JEPA_Encoder(proj_dim=proj_dim, imgformat='rgb', pretrained=pretrained).to(device)
    
    # Initialize target encoder as copy of context
    target_enc.load_state_dict(context_enc.state_dict())
    for p in target_enc.parameters():
        p.requires_grad = False  # frozen, updated by EMA
    
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


def SEGtrain(train_loader,
             valid_loader,
             device,   
             jepa_weights_path,
             weight_dir,
             modelname,
             proj_dim=256,
             out_classes=1,
             imgformat='rgb',
             lr=1e-3,
             epochs=10,
             patience=8):
    """
    Train a segmentation model (JEPAUNet) using a JEPA-pretrained encoder.

    Args:
        train_loader: DataLoader for training (image, mask)
        valid_loader: DataLoader for validation (image, mask)
        device: torch.device
        jepa_weights_path: path to trained JEPA encoder weights
        weight_dir: directory to save best decoder weights
        modelname: filename for saving decoder weights
        proj_dim: projection dimension (must match JEPA pretraining)
        out_classes: number of segmentation classes (1 for binary)
        imgformat: 'rgb' or 'gray'
        lr: learning rate
        epochs: number of training epochs
        patience: early stopping patience (epochs)
    """

    # ----------------------------
    # Initialize model with encoder
    # ----------------------------
    model = JEPAUNet(
        jepa_weights_path,
        proj_dim=proj_dim,
        out_classes=out_classes,
        imgformat=imgformat
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(model.decoder.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Loss
    criterion = nn.BCEWithLogitsLoss() if out_classes == 1 else nn.CrossEntropyLoss()

    # Best checkpoint tracking
    best_valid_dice = 0.0
    counter = 0

    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        start_time = time.time()

        # -------------------- TRAIN --------------------
        model.train()
        train_loss = 0.0
        for batch_idx, (imgs, masks) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            imgs = imgs.to(device, non_blocking=True).float() 
            masks = masks.to(device, non_blocking=True).float()

            # # Add channel dimension to mask if needed
            # if masks.ndim == 3:  # [B, H, W]
            #     masks = masks.unsqueeze(1)  # -> [B, 1, H, W]
        
            # Forward pass
            optimizer.zero_grad()    
            preds = model(imgs)  # forward pass
            
            # Backpropagation
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            print(f"\rBatch [{batch_idx+1}/{len(train_loader)}] --> Batch Loss: {loss.item():.6f}", end='')
            
        avg_train_loss = train_loss / len(train_loader)
        
        # -------------------- VALIDATION --------------------
        model.eval()
        valid_loss = 0.0
        dice_scores = []
        with torch.no_grad():
            for imgs, masks in valid_loader:
                imgs = imgs.to(device).float()
                masks = masks.to(device).float()
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)

                preds = model(imgs)
                loss = criterion(preds, masks)
                valid_loss += loss.item()

                # Compute Dice
                dice = calculate_dice(preds, masks)
                dice_scores.append(dice)

        avg_valid_loss = valid_loss / len(valid_loader)
        avg_valid_dice = sum(dice_scores) / len(dice_scores)
        
        # -------------------- CHECKPOINT --------------------
        if avg_valid_dice > best_valid_dice:
            best_valid_dice = avg_valid_dice
            torch.save(model.state_dict(), os.path.join(weight_dir, modelname))
            print(f"Model saved at epoch {epoch+1} with Dice: {avg_valid_dice:.4f}")
            counter = 0
        else:
            counter += 1
            print(f"No improvement in Dice. Early stopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping.")
                break

        scheduler.step()  # step LR scheduler
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_valid_loss:.4f}, "
              f"Val Dice: {avg_valid_dice:.4f}, "
              f"Time: {elapsed:.2f}s")

    print("\nSegmentation training completed.")
    print(f"Best Validation Dice Achieved: {best_valid_dice:.4f}")
