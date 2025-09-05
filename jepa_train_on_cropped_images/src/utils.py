import os
import cv2
import torch
import numpy as np
import PIL.Image as Image
import torch.nn.functional as F

def get_device_configuration():
    # Device configuration
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        print(f'Using cuda and device: "{torch.cuda.get_device_name(0)}"')
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("Using mps")
    else:
        DEVICE = torch.device("cpu")
        print("Using cpu")
    return DEVICE

def normalize_pixel_depth_values_to_rgb_array (img, floor_distance = 3000):

    # Convert PIL Image to numpy array if necessary
    if isinstance(img, Image.Image):
        img = np.array(img)

    
    img [img >= floor_distance] = 0
    valid_pixels = img > 0

    # Signal to skip this image 
    if not np.any(valid_pixels):
        return None
    
    depth_min, depth_max = np.min(img[valid_pixels]), np.max(img[valid_pixels])

    # Adjust depth_max slightly to avoid division by zero
    if depth_min == depth_max:
        depth_max += 1e-5

    depth_norm = np.zeros(img.shape, dtype=np.uint8)
    depth_norm[valid_pixels] = ((img[valid_pixels] - depth_min) / (depth_max - depth_min)*255 ).astype(np.uint8)

    depth_norm = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET) # convert to color img (RGB)
    depth_norm = cv2.cvtColor(depth_norm, cv2.COLOR_BGR2RGB)

    return Image.fromarray(depth_norm)


def calculate_dice(y_pred, y_true, smooth=1e-6, threshold=0.5):
    """
    Compute the Dice coefficient for binary segmentation.

    Args:
        y_pred (torch.Tensor): Predicted logits of shape (N, 1, H, W).
        y_true (torch.Tensor): Ground truth tensor of shape (N, 1, H, W) or (N, H, W).
        smooth (float): Small value to avoid division by zero.
        threshold (float): Threshold to binarize predictions.

    Returns:
        float: Mean Dice coefficient across the batch.
    """
    if y_true.ndim == 3:
        y_true = y_true.unsqueeze(1)

    if y_pred.shape[1] != 1:
        raise ValueError("y_pred must be of shape (N, 1, H, W) for binary segmentation.")

    # Apply sigmoid to logits and threshold
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > threshold).float()

    # Compute Dice for each sample in the batch
    dice_scores = []
    for i in range(y_pred.shape[0]):
        yp = y_pred[i].contiguous().view(-1)
        yt = y_true[i].contiguous().view(-1)
        intersection = (yp * yt).sum()
        union = yp.sum() + yt.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)

    return round(torch.stack(dice_scores).mean().item(), 2)

def calculate_iou(y_pred, y_true, smooth=1e-6, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) for binary segmentation.

    Args:
        y_pred (torch.Tensor): Predicted logits of shape (N, 1, H, W).
        y_true (torch.Tensor): Ground truth of shape (N, 1, H, W) or (N, H, W).
        smooth (float): Small value to avoid division by zero.
        threshold (float): Threshold for converting probabilities to binary.

    Returns:
        float: Mean IoU across the batch.
    """
    # Ensure ground truth shape
    if y_true.ndim == 3:
        y_true = y_true.unsqueeze(1)

    # Apply sigmoid and threshold
    if y_pred.shape[1] != 1:
        raise ValueError("y_pred must be of shape (N, 1, H, W) for binary segmentation.")
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > threshold).float()

    # Compute IoU for each image in the batch
    iou_scores = []
    for i in range(y_pred.shape[0]):
        yp = y_pred[i].contiguous().view(-1)
        yt = y_true[i].contiguous().view(-1)

        intersection = (yp * yt).sum()
        union = yp.sum() + yt.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou)

    return round(torch.stack(iou_scores).mean().item(),2)


def save_segmented_image(pred_mask, seg_dir, file_name, original_size):
    """
    Save the segmented image to the specified path.

    Args:
        pred_mask: The predicted mask tensor of shape (H, W) or (1, H, W).
        output_path: Path to save the image file.
        file_name: Original image file name
    """
    
    # Handle shape
    if pred_mask.dim() == 4:  # (B,C,H,W)
        if pred_mask.size(0) == 1 and pred_mask.size(1) == 1:
            pred_mask = pred_mask.squeeze(0).squeeze(0)
    elif pred_mask.dim() == 3:  # (C,H,W)
        if pred_mask.size(0) == 1:
            pred_mask = pred_mask.squeeze(0)

    # Keep as float until after resize
    pred_mask = pred_mask.detach().cpu()

    # Interpolate back to original size (note: size = (H,W))
    pred_mask = F.interpolate(
        pred_mask.unsqueeze(0).unsqueeze(0),
        size=(original_size[1], original_size[0]),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)

    # Threshold, still float
    pred_mask = (pred_mask > 0.5).float()

    # Convert to numpy and scale
    pred_mask_np = (pred_mask.numpy() * 255).astype(np.uint8)

    # Save as PNG
    pil_image = Image.fromarray(pred_mask_np)
    save_path = os.path.join(seg_dir, os.path.splitext(file_name)[0] + '.png')
    pil_image.save(save_path)
    
    
def summarize_results(results_df):
    """
    Generate statistical summary for Dice and IoU scores.

    Args:
        results_df: Dataframe containing the per-image Dice and IoU scores.

    Returns:
        dict: Summary statistics for Dice and IoU.
    """
    summary = {
        'Dice': {
            'Mean': results_df['Dice'].mean(),
            'Median': results_df['Dice'].median(),
            'StdDev': results_df['Dice'].std(),
            'Min': results_df['Dice'].min(),
            'Max': results_df['Dice'].max(),
        },
        'IoU': {
            'Mean': results_df['IoU'].mean(),
            'Median': results_df['IoU'].median(),
            'StdDev': results_df['IoU'].std(),
            'Min': results_df['IoU'].min(),
            'Max': results_df['IoU'].max(),
        },
    }

    # Print summary
    print("\nStatistical Summary:")
    for metric, stats in summary.items():
        print(f"\n{metric}:")
        for stat, value in stats.items():
            print(f"  {stat}: {value:.4f}")