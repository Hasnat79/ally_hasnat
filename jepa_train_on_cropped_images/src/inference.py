import pandas as pd
import torch
from utils import calculate_dice, calculate_iou
from utils import save_segmented_image

from model import JEPAUnet
def testing(jepa_unet_weights_path, data_loader, device, seg_dir):
    """
    Test the model on a dataset and compute Dice and IoU for each image.
    
    Args:
        jepa_unet_weights_path: The path to the JEPAUNet model trained.
        dataloader: DataLoader for the test dataset.
        device: The device to run the model on (e.g., 'cpu' or 'cuda').
        
    Returns:
        pd.DataFrame: Dataframe containing per-image results.
    """
    # Load trained decoder weights
    decoder = torch.load(jepa_unet_weights_path, map_location=device)
 
    # Build JEPAUNet with the same parameters as training
    model = JEPAUnet(
        jepa_unet_weights_path,     # path to the trained JEPA encoder
        proj_dim=256,               # match training
        out_classes=1,
        imgformat='rgb')
    
    model.load_state_dict(decoder)  # load trained decoder + encoder (if saved full model)
    model = model.to(device)
    model.eval()

    results = []  # Store results for each image  
    with torch.no_grad():
        for i, (image, mask, file_name, size) in enumerate(data_loader):
            image = image.to(device).float()
            mask = mask.to(device).float()
            
            # Forward pass
            outputs = model(image)
            
            # Compute metrics
            dice_score = calculate_dice(outputs, mask)
            iou_score = calculate_iou(outputs, mask)
            
            # Save segmented image 
            save_segmented_image(torch.sigmoid(outputs).squeeze(1), seg_dir, file_name[0], size)
            
            # Store the results
            results.append({'ImageID': file_name[0], 'Dice': dice_score, 'IoU': iou_score})
    
    # Return the results as a dataframe
    return pd.DataFrame(results)

def inference(model, dataloader, device, seg_dir):
    """
    Perform only inference.
    
    Args:
        model: The PyTorch model to test.
        dataloader: DataLoader for the test dataset.
        device: The device to run the model on (e.g., 'cpu' or 'cuda').
    
    Returns:
        Save segmented images in seg_dir
    """
    model.eval().to(device)  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient computation for testing
        for idx, (images, file_name, size) in enumerate(dataloader):
            images = images.float().to(device)
            
            # Forward pass
            outputs = model(images)
                        
            # Save segmented image 
            save_segmented_image(torch.sigmoid(outputs).squeeze(1), seg_dir, file_name[0], size)
