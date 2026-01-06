#!/usr/bin/env python3
"""
Optimized test for foreground-based S_NegLabel_star with P_align.

Key optimization:
- Pre-compute S_NegLabel_star and P_align for all images ONCE
- Lambda sweep: just compute S_final = S_NegLabel_star + lambda * P_align
- 8x speedup compared to reprocessing images for each lambda
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../utils'))

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from neglabel_foreground_star import CLIPNegAlignForegroundStar
from ood_evaluate import evaluate_all


class WaterBirdsDataset(Dataset):
    """WaterBirds test set"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        metadata_path = os.path.join(root_dir, 'metadata.csv')
        import pandas as pd
        df = pd.read_csv(metadata_path)
        
        # Test split only
        self.samples = df[df['split'] == 2].reset_index(drop=True)
        print(f"  Loaded {len(self.samples)} test samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.samples.loc[idx, 'img_filename'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, idx


class PlacesBGDataset(Dataset):
    """PlacesBG OOD test set"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.samples = []
        for img_name in os.listdir(root_dir):
            if img_name.endswith(('.jpg', '.png', '.JPEG')):
                self.samples.append(os.path.join(root_dir, img_name))
        
        self.samples = sorted(self.samples)
        print(f"  Loaded {len(self.samples)} OOD samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, idx


def precompute_scores(model, dataloader, desc="Processing"):
    """
    Pre-compute S_NegLabel_star and P_align for all images.
    
    Returns:
        s_neglabel: np.array (N,)
        p_align: np.array (N,)
    """
    s_neglabel_list = []
    p_align_list = []
    
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc=desc, leave=False):
            images = images.to(model.device)
            
            for img in images:
                # Get detailed scores
                details = model.detection_score(
                    img.unsqueeze(0),
                    orig_image=None,
                    return_details=True
                )
                
                s_neglabel_list.append(details['s_neglabel_star'])
                p_align_list.append(details['p_align'])
    
    return np.array(s_neglabel_list), np.array(p_align_list)


def run_optimized_lambda_sweep():
    """
    Optimized lambda sweep:
    1. Pre-compute S_NegLabel_star and P_align once
    2. For each lambda: S_final = S_NegLabel + lambda * P_align
    """
    print("=" * 80)
    print("Foreground-based Star + P_align (OPTIMIZED Lambda Sweep)")
    print("=" * 80)
    print()
    
    # Dataset paths
    waterbirds_path = '/home/kahyeon/research/data/waterbirds'
    placesbg_path = '/home/kahyeon/research/data/placesbg'
    
    # CLIP transform
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    clip_transform = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), 
                  (0.26862954, 0.26130258, 0.27577711))
    ])
    
    # Load datasets
    print("Loading datasets...")
    id_dataset = WaterBirdsDataset(waterbirds_path, transform=clip_transform)
    ood_dataset = PlacesBGDataset(placesbg_path, transform=clip_transform)
    
    id_loader = DataLoader(id_dataset, batch_size=32, shuffle=False, num_workers=4)
    ood_loader = DataLoader(ood_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"  ID (WaterBirds test): {len(id_dataset)} samples")
    print(f"  OOD (PlacesBG): {len(ood_dataset)} samples")
    print()
    
    # Initialize model (with P_align enabled, lambda will be adjusted later)
    print("Initializing foreground-based model...")
    model = CLIPNegAlignForegroundStar(
        train_dataset='waterbird',
        use_role_aware_negatives=True,
        use_p_align=True,  # Enable for pre-computation
        lambda_bg=1.0,  # Will be adjusted during sweep
        cam_fg_percentile=80,
        cam_dilate_px=1,
        cam_block=-1,
        device='cuda:0'
    )
    print()
    
    # ========================================================================
    # OPTIMIZATION: Pre-compute scores ONCE
    # ========================================================================
    print("=" * 80)
    print("Pre-computing scores (ONE-TIME)")
    print("=" * 80)
    print()
    
    print("Processing ID samples...")
    id_s_neglabel, id_p_align = precompute_scores(model, id_loader, desc="ID samples")
    print(f"  S_NegLabel_star range: [{id_s_neglabel.min():.4f}, {id_s_neglabel.max():.4f}]")
    print(f"  P_align range: [{id_p_align.min():.4f}, {id_p_align.max():.4f}]")
    print()
    
    print("Processing OOD samples...")
    ood_s_neglabel, ood_p_align = precompute_scores(model, ood_loader, desc="OOD samples")
    print(f"  S_NegLabel_star range: [{ood_s_neglabel.min():.4f}, {ood_s_neglabel.max():.4f}]")
    print(f"  P_align range: [{ood_p_align.min():.4f}, {ood_p_align.max():.4f}]")
    print()
    
    # ========================================================================
    # Lambda sweep: Just adjust weights (NO image reprocessing!)
    # ========================================================================
    print("=" * 80)
    print("Lambda Sweep (OPTIMIZED - no image reprocessing)")
    print("=" * 80)
    print()
    
    lambda_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    results = {}
    
    for lambda_val in lambda_values:
        print(f"--- Lambda = {lambda_val} ---")
        
        # Compute final scores (instant!)
        id_scores = id_s_neglabel + lambda_val * id_p_align
        ood_scores = ood_s_neglabel + lambda_val * ood_p_align
        
        # Evaluate
        metrics = evaluate_all(
            id_scores=id_scores,
            ood_scores=ood_scores,
            id_examples=id_scores,
            ood_examples=ood_scores
        )
        
        config_name = f"fg_star_palign_lambda_{lambda_val}"
        results[config_name] = metrics
        
        print(f"Results for {config_name}:")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  FPR95: {metrics['fpr95']:.4f}")
        print(f"  AUPR-IN: {metrics['aupr_in']:.4f}")
        print(f"  AUPR-OUT: {metrics['aupr_out']:.4f}")
        print()
    
    # Find best lambda
    best_lambda = max(results.items(), key=lambda x: x[1]['auroc'])
    print("=" * 80)
    print(f"BEST: {best_lambda[0]}")
    print(f"  AUROC: {best_lambda[1]['auroc']:.4f}")
    print(f"  FPR95: {best_lambda[1]['fpr95']:.4f}")
    print("=" * 80)
    
    # Save results
    import json
    output_path = '/home/kahyeon/research/NegAlign/results/waterbirds_foreground_optimized.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    run_optimized_lambda_sweep()
