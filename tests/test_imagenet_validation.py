#!/usr/bin/env python3
"""
ImageNet vs OOD Validation Test (Class-balanced sampling)

Quick validation experiment with balanced sampling across ImageNet classes.
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Add local utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from clip_negalign import CLIPNegAlign
from ood_evaluate import evaluate_all


def load_imagenet_balanced(root_dir, split='val', samples_per_class=5, seed=42):
    """
    Load ImageNet samples with balanced sampling across classes.
    
    Args:
        root_dir: Path to ImageNet directory (contains val/ or train/)
        split: 'val' or 'train'
        samples_per_class: Number of samples per class (for validation)
        seed: Random seed for reproducibility
    
    Returns:
        List of sample dicts with 'path' and 'label'
    """
    split_dir = Path(root_dir) / split
    
    if not split_dir.exists():
        raise FileNotFoundError(f"ImageNet {split} directory not found: {split_dir}")
    
    # Collect all class directories (synsets)
    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} ImageNet classes")
    
    # Group images by class
    class_to_images = defaultdict(list)
    
    for class_idx, class_dir in enumerate(tqdm(class_dirs, desc="Scanning classes")):
        # Get all images in this class
        for ext in ['*.JPEG', '*.jpg', '*.png']:
            for img_path in class_dir.glob(ext):
                class_to_images[class_idx].append({
                    'path': str(img_path),
                    'label': class_idx,
                    'class_name': class_dir.name
                })
    
    # Sample uniformly from each class
    np.random.seed(seed)
    samples = []
    
    for class_idx in range(len(class_dirs)):
        class_images = class_to_images[class_idx]
        
        if len(class_images) == 0:
            print(f"Warning: No images found for class {class_idx}")
            continue
        
        # Sample with replacement if needed
        n_samples = min(samples_per_class, len(class_images))
        sampled_indices = np.random.choice(len(class_images), n_samples, replace=False)
        
        for idx in sampled_indices:
            samples.append(class_images[idx])
    
    print(f"Sampled {len(samples)} images ({samples_per_class} per class x {len(class_dirs)} classes)")
    
    return samples


def load_ood_dataset(name, root_dir, max_samples=None, seed=42):
    """
    Load OOD dataset samples.
    
    Args:
        name: Dataset name ('inaturalist', 'sun', 'places', 'texture', 'ninco')
        root_dir: Root directory containing OOD datasets
        max_samples: Maximum number of samples (optional)
        seed: Random seed for sampling
    """
    dataset_paths = {
        'inaturalist': 'iNaturalist/images',
        'sun': 'SUN', 
        'places': 'Places/val_256',
        'texture': 'dtd/images',
        'ninco': 'NINCO/NINCO_OOD_classes'
    }
    
    if name not in dataset_paths:
        raise ValueError(f"Unknown OOD dataset: {name}")
    
    ood_path = Path(root_dir) / dataset_paths[name]
    
    if not ood_path.exists():
        raise FileNotFoundError(f"OOD dataset not found: {ood_path}")
    
    # Collect all images recursively
    samples = []
    for ext in ['.JPEG', '.jpg', '.png', '.jpeg']:
        for img_path in ood_path.rglob(f'*{ext}'):
            samples.append({
                'path': str(img_path),
                'label': -1  # OOD label
            })
    
    # Random sample if max_samples specified
    if max_samples is not None and max_samples < len(samples):
        np.random.seed(seed)
        indices = np.random.choice(len(samples), max_samples, replace=False)
        samples = [samples[i] for i in sorted(indices)]
    
    print(f"OOD ({name}): {len(samples)} samples")
    
    return samples


def main():
    parser = argparse.ArgumentParser(description='ImageNet vs OOD Validation Test')
    parser.add_argument('--imagenet_root', type=str,
                       default='/home/kahyeon/datasets/imagenet',
                       help='Path to ImageNet dataset')
    parser.add_argument('--ood_root', type=str,
                       default='/home/kahyeon/datasets',
                       help='Root path to OOD datasets')
    parser.add_argument('--ood_datasets', type=str, nargs='+',
                       default=['inaturalist', 'ninco', 'sun', 'texture'],
                       help='OOD datasets to test')
    parser.add_argument('--samples_per_class', type=int, default=5,
                       help='Number of samples per ImageNet class')
    parser.add_argument('--max_ood_samples', type=int, default=5000,
                       help='Maximum OOD samples per dataset')
    parser.add_argument('--output_dir', type=str, default='./results/imagenet_validation',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='CUDA device')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("ImageNet vs OOD Validation Test (Class-balanced)")
    print("="*80)
    print(f"ImageNet: {args.samples_per_class} samples/class x 1000 classes = {args.samples_per_class * 1000} total")
    print(f"OOD datasets: {', '.join(args.ood_datasets)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")

    # Initialize model
    print("\n" + "="*80)
    print("Initializing NegAlign model...")
    print("="*80)

    model = CLIPNegAlign(
        train_dataset='imagenet',
        arch='ViT-B/16',
        seed=args.seed,
        device=args.device,
        output_folder='./data/',
        load_saved_labels=True,
        # Model settings
        use_neglabel_star=False,  # Use plain NegLabel
        use_role_aware_negatives=True,
        use_p_align=True,
        lambda_bg=1.0,
        pos_topk=10,
        neg_topk=5,
        cam_fg_percentile=80,
        cam_dilate_px=1,
        cam_block=-1
    )

    # Load ImageNet samples (class-balanced)
    print("\n" + "="*80)
    print("Loading ImageNet validation samples...")
    print("="*80)

    id_samples = load_imagenet_balanced(
        args.imagenet_root,
        split='val',
        samples_per_class=args.samples_per_class,
        seed=args.seed
    )

    # Process ID samples
    print("\n" + "="*80)
    print("Processing ID samples (ImageNet)...")
    print("="*80)

    id_scores_plain = []
    id_scores_star = []
    id_p_align = []
    
    for sample in tqdm(id_samples, desc="ID"):
        img_path = sample['path']
        img = Image.open(img_path).convert('RGB')
        img_tensor = model.clip_preprocess(img).unsqueeze(0).to(model.device)

        # Get detection scores with details
        result = model.detection_score(img_tensor, orig_image=img, return_details=True)
        id_scores_plain.append(result['s_neglabel_plain'])
        id_scores_star.append(result['s_neglabel_star'])
        id_p_align.append(result['p_align'])

    id_scores_plain = np.array(id_scores_plain)
    id_scores_star = np.array(id_scores_star)
    id_p_align = np.array(id_p_align)
    
    print(f"\nID Statistics:")
    print(f"  S_NegLabel_plain (6370 negs): mean={id_scores_plain.mean():.4f}, std={id_scores_plain.std():.4f}")
    print(f"  S_NegLabel_star (2866 negs):  mean={id_scores_star.mean():.4f}, std={id_scores_star.std():.4f}")
    print(f"  P_align:                      mean={id_p_align.mean():.4f}, std={id_p_align.std():.4f}")

    # Lambda values for grid search
    lambda_values = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    
    # Test each OOD dataset
    all_results = {
        'config': {
            'samples_per_class': args.samples_per_class,
            'max_ood_samples': args.max_ood_samples,
            'seed': args.seed,
            'lambda_values': lambda_values
        },
        'id_stats': {
            'dataset': 'imagenet',
            'n_samples': len(id_samples),
            's_plain_mean': float(id_scores_plain.mean()),
            's_star_mean': float(id_scores_star.mean()),
            'p_align_mean': float(id_p_align.mean())
        },
        'ood_results': {}
    }

    for ood_name in args.ood_datasets:
        print("\n" + "="*80)
        print(f"Testing OOD: {ood_name.upper()}")
        print("="*80)

        # Load OOD samples
        try:
            ood_samples = load_ood_dataset(
                ood_name,
                args.ood_root,
                max_samples=args.max_ood_samples,
                seed=args.seed
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading {ood_name}: {e}")
            continue
        
        # Process OOD samples
        ood_scores_plain = []
        ood_scores_star = []
        ood_p_align = []
        
        for sample in tqdm(ood_samples, desc=f"OOD ({ood_name})"):
            img_path = sample['path']
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = model.clip_preprocess(img).unsqueeze(0).to(model.device)
                
                result = model.detection_score(img_tensor, orig_image=img, return_details=True)
                ood_scores_plain.append(result['s_neglabel_plain'])
                ood_scores_star.append(result['s_neglabel_star'])
                ood_p_align.append(result['p_align'])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        ood_scores_plain = np.array(ood_scores_plain)
        ood_scores_star = np.array(ood_scores_star)
        ood_p_align = np.array(ood_p_align)
        
        print(f"\nOOD Statistics ({ood_name}):")
        print(f"  S_NegLabel_plain (6370 negs): mean={ood_scores_plain.mean():.4f}, std={ood_scores_plain.std():.4f}")
        print(f"  S_NegLabel_star (2866 negs):  mean={ood_scores_star.mean():.4f}, std={ood_scores_star.std():.4f}")
        print(f"  P_align:                      mean={ood_p_align.mean():.4f}, std={ood_p_align.std():.4f}")

        # ========================================================
        # [BASELINE 1] Plain NegLabel (6370 negatives)
        # ========================================================
        print(f"\n{'='*80}")
        print(f"[BASELINE 1] Plain NegLabel (all 6370 negatives)")
        print(f"{'='*80}")
        auroc_plain, aupr_in_plain, aupr_out_plain, fpr95_plain = evaluate_all(id_scores_plain, ood_scores_plain)
        print(f"  AUROC:    {auroc_plain:.4f}")
        print(f"  FPR95:    {fpr95_plain:.4f}")
        print(f"  AUPR-IN:  {aupr_in_plain:.4f}")
        print(f"  AUPR-OUT: {aupr_out_plain:.4f}")

        # ========================================================
        # [BASELINE 2] Star NegLabel (2866 N_obj only)
        # ========================================================
        print(f"\n{'='*80}")
        print(f"[BASELINE 2] Star NegLabel (N_obj 2866 only)")
        print(f"{'='*80}")
        auroc_star, aupr_in_star, aupr_out_star, fpr95_star = evaluate_all(id_scores_star, ood_scores_star)
        print(f"  AUROC:    {auroc_star:.4f}")
        print(f"  FPR95:    {fpr95_star:.4f}")
        print(f"  AUPR-IN:  {aupr_in_star:.4f}")
        print(f"  AUPR-OUT: {aupr_out_star:.4f}")

        # ========================================================
        # [METHOD 1] Plain (6370) + λ*P_align
        # ========================================================
        print(f"\n{'='*80}")
        print(f"[METHOD 1] Plain (6370) + λ*P_align")
        print(f"{'='*80}")
        print(f"{'Lambda':<8} {'AUROC':<8} {'FPR95':<8} {'AUPR-IN':<10} {'AUPR-OUT':<10}")
        print(f"{'='*80}")
        
        plain_lambda_results = []
        best_plain_auroc = 0.0
        best_plain_lambda = 0.0
        
        for lam in lambda_values:
            id_combined = id_scores_plain + lam * id_p_align
            ood_combined = ood_scores_plain + lam * ood_p_align
            auroc, aupr_in, aupr_out, fpr95 = evaluate_all(id_combined, ood_combined)
            
            plain_lambda_results.append({
                'lambda': float(lam),
                'auroc': float(auroc),
                'fpr95': float(fpr95),
                'aupr_in': float(aupr_in),
                'aupr_out': float(aupr_out)
            })
            
            print(f"{lam:<8.1f} {auroc:<8.4f} {fpr95:<8.4f} {aupr_in:<10.4f} {aupr_out:<10.4f}")
            
            if auroc > best_plain_auroc:
                best_plain_auroc = auroc
                best_plain_lambda = lam
        
        print(f"{'='*80}")
        print(f"Best Lambda: {best_plain_lambda}, AUROC: {best_plain_auroc:.4f}")

        # ========================================================
        # [METHOD 2] Star (2866) + λ*P_align
        # ========================================================
        print(f"\n{'='*80}")
        print(f"[METHOD 2] Star (2866) + λ*P_align")
        print(f"{'='*80}")
        print(f"{'Lambda':<8} {'AUROC':<8} {'FPR95':<8} {'AUPR-IN':<10} {'AUPR-OUT':<10}")
        print(f"{'='*80}")
        
        star_lambda_results = []
        best_star_auroc = 0.0
        best_star_lambda = 0.0
        
        for lam in lambda_values:
            # Compute combined scores: S_star + lambda * P_align
            id_combined = id_scores_star + lam * id_p_align
            ood_combined = ood_scores_star + lam * ood_p_align
            
            # Evaluate
            auroc, aupr_in, aupr_out, fpr95 = evaluate_all(id_combined, ood_combined)
            
            star_lambda_results.append({
                'lambda': float(lam),
                'auroc': float(auroc),
                'fpr95': float(fpr95),
                'aupr_in': float(aupr_in),
                'aupr_out': float(aupr_out)
            })
            
            print(f"{lam:<8.1f} {auroc:<8.4f} {fpr95:<8.4f} {aupr_in:<10.4f} {aupr_out:<10.4f}")
            
            if auroc > best_star_auroc:
                best_star_auroc = auroc
                best_star_lambda = lam
        
        print(f"{'='*80}")
        print(f"Best Lambda: {best_star_lambda}, AUROC: {best_star_auroc:.4f}")

        # Store all 4 method results
        all_results['ood_results'][ood_name] = {
            'n_samples': len(ood_samples),
            's_plain_mean': float(ood_scores_plain.mean()),
            's_star_mean': float(ood_scores_star.mean()),
            'p_align_mean': float(ood_p_align.mean()),
            'baseline_plain': {
                'auroc': float(auroc_plain),
                'fpr95': float(fpr95_plain),
                'aupr_in': float(aupr_in_plain),
                'aupr_out': float(aupr_out_plain)
            },
            'baseline_star': {
                'auroc': float(auroc_star),
                'fpr95': float(fpr95_star),
                'aupr_in': float(aupr_in_star),
                'aupr_out': float(aupr_out_star)
            },
            'plain_plus_p_align': {
                'lambda_sweep': plain_lambda_results,
                'best_lambda': float(best_plain_lambda),
                'best_auroc': float(best_plain_auroc)
            },
            'star_plus_p_align': {
                'lambda_sweep': star_lambda_results,
                'best_lambda': float(best_star_lambda),
                'best_auroc': float(best_star_auroc)
            }
        }

    # Save results
    output_file = Path(args.output_dir) / 'validation_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("SUMMARY: ImageNet vs OOD Validation Results")
    print("="*80)
    print(f"\nID: ImageNet ({len(id_samples)} samples, {args.samples_per_class} per class)")
    print(f"  S_plain: {id_scores_plain.mean():.4f}, S_star: {id_scores_star.mean():.4f}, P_align: {id_p_align.mean():.4f}")
    
    for ood_name, result in all_results['ood_results'].items():
        print(f"\n{'='*80}")
        print(f"OOD: {ood_name.upper()} ({result['n_samples']} samples)")
        print(f"{'='*80}")
        print(f"  [1] Plain (6370):       AUROC={result['baseline_plain']['auroc']:.4f}, FPR95={result['baseline_plain']['fpr95']:.4f}")
        print(f"  [2] Star (2866):        AUROC={result['baseline_star']['auroc']:.4f}, FPR95={result['baseline_star']['fpr95']:.4f}")
        print(f"  [3] Plain + P_align:    Best λ={result['plain_plus_p_align']['best_lambda']:.1f}, AUROC={result['plain_plus_p_align']['best_auroc']:.4f}")
        print(f"  [4] Star + P_align:     Best λ={result['star_plus_p_align']['best_lambda']:.1f}, AUROC={result['star_plus_p_align']['best_auroc']:.4f}")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

