# -*- coding: utf-8 -*-
# @Author : Xuanhe Er
# @Time   : 28/07/2023 12:10

import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import InterpolationMode
import random

from math import exp
import torch.nn.functional as F

from dataset import bcidataset
#from dataset import mistdataset

def pil_loader(path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def plot_images(images):
    plt.figure(figsize=(16, 16))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def save_images(images, path, nrow=10, **kwargs):
    grid = torchvision.utils.make_grid(images, nrow=nrow, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def save_single_image(image, path):
    image_numpy = image.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(image_numpy)
    im.save(path)    
    
def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.image_size),  # Use PIL's built-in LANCZOS or BOX  , interpolation=Image.LANCZOS
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = bcidataset.BCIDataset(x_img_dir=args.x_image_dir_path, y_img_dir=args.y_image_dir_path,
                                    num_pairs=args.num_pairs, transforms=transforms)

    # Use MISTDataset instead of BCIDataset
    # dataset = mistdataset.MISTDataset(x_img_dir=args.x_image_dir_path, y_img_dir=args.y_image_dir_path,
    #                                  num_pairs=args.num_pairs, transforms=transforms)

    if args.use_multi_gpu:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=min(8, os.cpu_count()), pin_memory=True)
    else:
        #dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    return dataloader

# --------------------------------------------------------------------------------------------------------

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def SSIM(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs

    return ret

# ---------------------------------------------------------------------------------------

def SSIM_decomposed(img1, img2, val_range, window_size=11, window=None, size_average=True, return_maps=False):
    """
    Returns decomposed SSIM components: 
    - LC (Luminance-Contrast)
    - SS (Source-Structural)
    
    Also returns full SSIM for backward compatibility
    """
    L = val_range
    padd = 0
    (_, channel, height, width) = img1.size()
    
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    # Local means
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Variances and covariances
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    # Clamp variances to avoid negative values
    sigma1_sq = torch.clamp(sigma1_sq, min=0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0)
    
    # Standard deviations
    sigma1 = torch.sqrt(sigma1_sq)
    sigma2 = torch.sqrt(sigma2_sq)

    # SSIM constants
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    C3 = C2 / 2  # For structural component

    # ====== Decomposed Components ======
    # 1. Luminance Component
    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    
    # 2. Contrast Component
    contrast = (2 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
    
    # 3. LC = Luminance × Contrast
    lc_map = luminance * contrast
    
    # 4. SS (Structural Similarity)
    ss_map = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    
    # ====== Full SSIM ======
    ssim_map = lc_map * ss_map

    # Average if needed
    if size_average:
        lc = lc_map.mean() #.item()
        ss = ss_map.mean() #.item()
        full_ssim = ssim_map.mean() #.item()
    else:
        lc = lc_map.mean(1).mean(1).mean(1) #.item()
        ss = ss_map.mean(1).mean(1).mean(1) #.item()
        full_ssim = ssim_map.mean(1).mean(1).mean(1) #.item()

    if return_maps:
        return full_ssim, lc, ss, ssim_map, lc_map, ss_map
    else:
        return full_ssim, lc, ss

# --------------------------------------------------------------------------------

def CCS(gt_img, gen_img, k=1e-6):
    """
    Compute Color Consistency Similarity (CCS) between two images using vectorized operations.
    
    Args:
        gt_img: Ground truth image (numpy array, uint8, [0, 255])
        gen_img: Generated image (numpy array, uint8, [0, 255])
        k: Small constant for numerical stability
        
    Returns:
        ccs: Color Consistency Similarity score (higher is better)
    """
    # Convert to XYZ color space
    gt_xyz = rgb_to_xyz(gt_img)
    gen_xyz = rgb_to_xyz(gen_img)
    
    # Compute squared differences in one operation
    delta_squared = (gt_xyz - gen_xyz) ** 2
    
    # Compute L color map using vectorized operations
    L_map = 0.4 * np.sqrt(np.sum(delta_squared, axis=2) + k)
    
    # Compute average L and convert to similarity score
    average_L = np.mean(L_map)
    return 1 / (1 + average_L)

def rgb_to_xyz(rgb_image):
    """
    Vectorized RGB to XYZ conversion for entire images.
    
    Args:
        rgb_image: numpy array of shape (H, W, 3) in [0, 255] range
        
    Returns:
        xyz_image: numpy array of shape (H, W, 3) in XYZ color space
    """
    # Normalize to [0, 1] and reshape to (H*W, 3)
    rgb = rgb_image.astype(np.float32) / 255.0
    orig_shape = rgb.shape
    pixels = rgb.reshape(-1, 3)
    
    # Define conversion matrix (with the 1/0.47697 factor)
    M = np.array([
        [0.49000, 0.3100, 0.20000],
        [0.17697, 0.8124, 0.01063],
        [0.00000, 0.0100, 0.99000]
    ], dtype=np.float32) / 0.17697
    
    # Vectorized matrix multiplication
    xyz_pixels = np.dot(pixels, M.T)
    
    # Reshape back to original dimensions
    return xyz_pixels.reshape(orig_shape)

# ============================================================================================================================
def count_parameters(model):
    """Minimal parameter counter without table"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total parameters: {total/1e6:.3f} M | Trainable: {trainable/1e6:.3f} M"

# ============================================================================================================================

class MultiScaleL1Loss(nn.Module):
    def __init__(self, scales=3, weights=None, reduction='mean'):
        """
        Multi-scale L1 loss for Gaussian noise prediction
        
        Args:
            scales (int): Number of downscaling levels
            weights (list): Weight for each scale (default: equal weight)
            reduction (str): 'mean' or 'sum'
        """
        super().__init__()
        self.scales = scales
        self.reduction = reduction
        self.weights = weights or [1.0/scales] * scales
        
    def forward(self, pred, target):
        total_loss = 0.0
        current_pred, current_target = pred, target
        
        for i in range(self.scales):
            # Compute L1 at current scale
            scale_loss = F.l1_loss(current_pred, current_target, reduction=self.reduction)
            
            # Apply scale weighting
            if self.reduction != 'none':
                total_loss += self.weights[i] * scale_loss
            
            # Downsample for next scale (skip on last iteration)
            if i < self.scales - 1:
                current_pred = F.avg_pool2d(current_pred, kernel_size=2, stride=2)
                current_target = F.avg_pool2d(current_target, kernel_size=2, stride=2)
                
        return total_loss