import cv2
import torch
import numpy as np
import PIL.Image as Image


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