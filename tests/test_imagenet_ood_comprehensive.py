#!/usr/bin/env python3
"""
Comprehensive ImageNet OOD test with same conditions as WaterBirds:
1. Baseline (plain, no amb, λ=0)
2. Soft-only (star+weighted, λ=0)
3. Soft + P_align (optimized, λ=1.0~10.0)
4. Foreground-based star + P_align (optimized, λ=0.0~10.0)

OOD datasets: iNaturalist, NINCO, SUN, DTD
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from core.clip_negalign import CLIPNegAlign
from core.neglabel_soft_assign import CLIPNegAlignSoftAssign
from core.neglabel_foreground_star import CLIPNegAlignForegroundStar
from utils.ood_evaluate import evaluate_all


def load_image(img_path):
    """Load and preprocess image for CLIP."""
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    
    clip_transform = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), 
                  (0.26862954, 0.26130258, 0.27577711))
    ])
    
    image = Image.open(img_path).convert('RGB')
    return clip_transform(image), image


def load_imagenet_val_samples():
    """Load ImageNet validation set as ID (5 samples per class = 5000 total)."""
    imagenet_root = '/home/kahyeon/datasets/imagenet/val'
    samples = []
    samples_per_class = 5
    
    for class_dir in sorted(os.listdir(imagenet_root)):
        class_path = os.path.join(imagenet_root, class_dir)
        if not os.path.isdir(class_path):
            continue
        
        class_samples = []
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_path, img_file)
                class_samples.append({'path': img_path, 'label': class_dir})
        
        # Take only first 5 samples per class
        samples.extend(class_samples[:samples_per_class])
    
    print(f"  Loaded {len(samples)} ImageNet validation samples ({samples_per_class} per class)")
    return samples


def load_ood_samples(ood_name, max_samples=5000):
    """Load OOD dataset samples (limited to max_samples)."""
    ood_roots = {
        'inaturalist': '/home/kahyeon/datasets/iNaturalist',
        'ninco': '/home/kahyeon/datasets/NINCO/NINCO_OOD_unit_tests',
        'sun': '/nas_homes/dataset/sun397/SUN397',
        'dtd': '/home/kahyeon/datasets/dtd/images'
    }
    
    root = ood_roots[ood_name.lower()]
    samples = []
    
    # Recursively find all images
    for dirpath, dirnames, filenames in os.walk(root):
        for img_file in filenames:
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(dirpath, img_file)
                samples.append({'path': img_path, 'label': -1})
                
                # Stop if we reached max_samples
                if len(samples) >= max_samples:
                    break
        if len(samples) >= max_samples:
            break
    
    print(f"  Loaded {len(samples)} {ood_name} samples")
    return samples


def evaluate_method(model, id_samples, ood_samples, method_name):
    """Evaluate a method on ID and OOD samples."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {method_name}")
    print(f"{'='*80}")
    
    # ID scores
    print("Processing ID samples...")
    id_scores = []
    for i, sample in enumerate(id_samples):
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{len(id_samples)} ID samples")
        
        img, _ = load_image(sample['path'])
        img = img.to(model.device)
        score = model.detection_score(img.unsqueeze(0))
        id_scores.append(float(score))
    
    # OOD scores
    print("\nProcessing OOD samples...")
    ood_scores = []
    for i, sample in enumerate(ood_samples):
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{len(ood_samples)} OOD samples")
        
        img, _ = load_image(sample['path'])
        img = img.to(model.device)
        score = model.detection_score(img.unsqueeze(0))
        ood_scores.append(float(score))
    
    # Evaluate
    print("\nComputing metrics...")
    id_scores = np.array(id_scores)
    ood_scores = np.array(ood_scores)
    
    print(f"  ID score range: [{id_scores.min():.4f}, {id_scores.max():.4f}], mean={id_scores.mean():.4f}")
    print(f"  OOD score range: [{ood_scores.min():.4f}, {ood_scores.max():.4f}], mean={ood_scores.mean():.4f}")
    
    metrics = evaluate_all(in_scores=id_scores, out_scores=ood_scores)
    
    print(f"\nResults for {method_name}:")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    print(f"  FPR95: {metrics['fpr95']:.4f}")
    print(f"  AUPR-IN: {metrics['aupr_in']:.4f}")
    print(f"  AUPR-OUT: {metrics['aupr_out']:.4f}")
    
    return {
        'method': method_name,
        'metrics': metrics,
        'id_scores': id_scores.tolist(),
        'ood_scores': ood_scores.tolist()
    }


def main():
    print("="*80)
    print("ImageNet OOD Comprehensive Test")
    print("="*80)
    print()
    
    # Load ID dataset
    print("Loading ID dataset (ImageNet val)...")
    id_samples = load_imagenet_val_samples()
    
    # OOD datasets
    ood_datasets = ['iNaturalist', 'NINCO', 'SUN', 'DTD']
    
    # Results storage
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'id_dataset': 'ImageNet_val',
        'id_samples': len(id_samples),
        'ood_datasets': {}
    }
    
    # Test each OOD dataset
    for ood_name in ood_datasets:
        print(f"\n{'='*80}")
        print(f"OOD Dataset: {ood_name}")
        print(f"{'='*80}")
        
        # Load OOD samples
        print(f"Loading OOD dataset ({ood_name})...")
        ood_samples = load_ood_samples(ood_name)        
        # Skip if no samples loaded
        if len(ood_samples) == 0:
            print(f"  WARNING: No samples loaded for {ood_name}, skipping...")
            continue        
        ood_results = {
            'ood_samples': len(ood_samples),
            'experiments': {}
        }
        
        # ====================================================================
        # 1. Baseline: plain, no amb, λ=0
        # ====================================================================
        print(f"\n{'='*80}")
        print(f"1. Baseline: use_amb_soft=False, lambda=0")
        print(f"{'='*80}")
        
        model_baseline = CLIPNegAlign(
            train_dataset='imagenet',
            arch='ViT-B/16',
            seed=0,
            device='cuda:0',
            output_folder='./data/',
            load_saved_labels=True,
            use_neglabel_star=True,
            use_role_aware_negatives=True,
            use_p_align=False,
            lambda_bg=0.0
        )
        
        result_baseline = evaluate_method(
            model_baseline,
            id_samples,
            ood_samples,
            'baseline_no_amb'
        )
        ood_results['experiments']['baseline_no_amb'] = result_baseline
        
        del model_baseline
        torch.cuda.empty_cache()
        
        # ====================================================================
        # 2. Soft assignment only: star+weighted, λ=0
        # ====================================================================
        print(f"\n{'='*80}")
        print(f"2. Soft Assignment Only: use_amb_soft=True, lambda=0")
        print(f"{'='*80}")
        
        model_soft = CLIPNegAlignSoftAssign(
            train_dataset='imagenet',
            arch='ViT-B/16',
            seed=0,
            device='cuda:0',
            output_folder='./data/',
            load_saved_labels=True,
            use_neglabel_star=True,
            use_role_aware_negatives=True,
            use_p_align=False,
            lambda_bg=0.0,
            use_amb_soft=True,
            amb_soft_beta=10.0
        )
        
        result_soft = evaluate_method(
            model_soft,
            id_samples,
            ood_samples,
            'soft_only'
        )
        ood_results['experiments']['soft_only'] = result_soft
        
        del model_soft
        torch.cuda.empty_cache()
        
        # ====================================================================
        # 3. Soft + P_align: OPTIMIZED Lambda sweep
        # ====================================================================
        print(f"\n{'='*80}")
        print(f"3. Soft Assignment + P_align: OPTIMIZED Lambda sweep")
        print(f"{'='*80}")
        print("(Pre-computes scores once, then adjusts lambda)\n")
        
        print("Initializing soft assignment model with P_align...")
        model_soft_palign = CLIPNegAlignSoftAssign(
            train_dataset='imagenet',
            arch='ViT-B/16',
            seed=0,
            device='cuda:0',
            output_folder='./data/',
            load_saved_labels=True,
            use_neglabel_star=True,
            use_role_aware_negatives=True,
            use_p_align=True,
            lambda_bg=1.0,
            use_amb_soft=True,
            amb_soft_beta=10.0
        )
        
        # Pre-compute scores
        print("\nPre-computing scores (ONE-TIME)...")
        print("Processing ID samples...")
        id_s_neglabel_soft = []
        id_p_align_soft = []
        for i, sample in enumerate(id_samples):
            if (i + 1) % 5000 == 0:
                print(f"  Processed {i+1}/{len(id_samples)}")
            img, _ = load_image(sample['path'])
            img = img.to(model_soft_palign.device)
            details = model_soft_palign.detection_score(img.unsqueeze(0), orig_image=None, return_details=True)
            id_s_neglabel_soft.append(details['s_neglabel_star'])
            id_p_align_soft.append(details['p_align'])
        
        id_s_neglabel_soft = np.array(id_s_neglabel_soft)
        id_p_align_soft = np.array(id_p_align_soft)
        print(f"  S_NegLabel_star range: [{id_s_neglabel_soft.min():.4f}, {id_s_neglabel_soft.max():.4f}]")
        print(f"  P_align range: [{id_p_align_soft.min():.4f}, {id_p_align_soft.max():.4f}]")
        
        print("\nProcessing OOD samples...")
        ood_s_neglabel_soft = []
        ood_p_align_soft = []
        for i, sample in enumerate(ood_samples):
            if (i + 1) % 5000 == 0:
                print(f"  Processed {i+1}/{len(ood_samples)}")
            img, _ = load_image(sample['path'])
            img = img.to(model_soft_palign.device)
            details = model_soft_palign.detection_score(img.unsqueeze(0), orig_image=None, return_details=True)
            ood_s_neglabel_soft.append(details['s_neglabel_star'])
            ood_p_align_soft.append(details['p_align'])
        
        ood_s_neglabel_soft = np.array(ood_s_neglabel_soft)
        ood_p_align_soft = np.array(ood_p_align_soft)
        print(f"  S_NegLabel_star range: [{ood_s_neglabel_soft.min():.4f}, {ood_s_neglabel_soft.max():.4f}]")
        print(f"  P_align range: [{ood_p_align_soft.min():.4f}, {ood_p_align_soft.max():.4f}]")
        
        # Lambda sweep
        print("\nLambda sweep (OPTIMIZED)...")
        lambda_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        for lambda_bg in lambda_values:
            print(f"\n--- Lambda = {lambda_bg} ---")
            id_scores = id_s_neglabel_soft + lambda_bg * id_p_align_soft
            ood_scores = ood_s_neglabel_soft + lambda_bg * ood_p_align_soft
            
            metrics = evaluate_all(in_scores=id_scores, out_scores=ood_scores)
            
            ood_results['experiments'][f'soft_palign_lambda_{lambda_bg}'] = {
                'method': f'soft_palign_lambda_{lambda_bg}',
                'metrics': metrics
            }
            
            print(f"  AUROC: {metrics['auroc']:.4f}, FPR95: {metrics['fpr95']:.4f}")
        
        del model_soft_palign
        torch.cuda.empty_cache()
        
        # ====================================================================
        # 4. Foreground-based Star + P_align: OPTIMIZED Lambda sweep
        # ====================================================================
        print(f"\n{'='*80}")
        print(f"4. Foreground-based Star + P_align: OPTIMIZED Lambda sweep")
        print(f"{'='*80}")
        print("(Pre-computes scores once, then adjusts lambda)\n")
        
        print("Initializing foreground-based model...")
        model_fg = CLIPNegAlignForegroundStar(
            train_dataset='imagenet',
            arch='ViT-B/16',
            seed=0,
            device='cuda:0',
            output_folder='./data/',
            load_saved_labels=True,
            use_role_aware_negatives=True,
            use_p_align=True,
            lambda_bg=1.0,
            cam_fg_percentile=80,
            cam_dilate_px=1,
            cam_block=-1
        )
        
        # Pre-compute scores
        print("\nPre-computing scores (ONE-TIME)...")
        print("Processing ID samples...")
        id_s_neglabel = []
        id_p_align = []
        for i, sample in enumerate(id_samples):
            if (i + 1) % 5000 == 0:
                print(f"  Processed {i+1}/{len(id_samples)}")
            img, _ = load_image(sample['path'])
            img = img.to(model_fg.device)
            details = model_fg.detection_score(img.unsqueeze(0), orig_image=None, return_details=True)
            id_s_neglabel.append(details['s_neglabel_star'])
            id_p_align.append(details['p_align'])
        
        id_s_neglabel = np.array(id_s_neglabel)
        id_p_align = np.array(id_p_align)
        print(f"  S_NegLabel_star range: [{id_s_neglabel.min():.4f}, {id_s_neglabel.max():.4f}]")
        print(f"  P_align range: [{id_p_align.min():.4f}, {id_p_align.max():.4f}]")
        
        print("\nProcessing OOD samples...")
        ood_s_neglabel = []
        ood_p_align = []
        for i, sample in enumerate(ood_samples):
            if (i + 1) % 5000 == 0:
                print(f"  Processed {i+1}/{len(ood_samples)}")
            img, _ = load_image(sample['path'])
            img = img.to(model_fg.device)
            details = model_fg.detection_score(img.unsqueeze(0), orig_image=None, return_details=True)
            ood_s_neglabel.append(details['s_neglabel_star'])
            ood_p_align.append(details['p_align'])
        
        ood_s_neglabel = np.array(ood_s_neglabel)
        ood_p_align = np.array(ood_p_align)
        print(f"  S_NegLabel_star range: [{ood_s_neglabel.min():.4f}, {ood_s_neglabel.max():.4f}]")
        print(f"  P_align range: [{ood_p_align.min():.4f}, {ood_p_align.max():.4f}]")
        
        # Lambda sweep
        print("\nLambda sweep (OPTIMIZED)...")
        lambda_values_fg = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        for lambda_bg in lambda_values_fg:
            print(f"\n--- Lambda = {lambda_bg} ---")
            id_scores = id_s_neglabel + lambda_bg * id_p_align
            ood_scores = ood_s_neglabel + lambda_bg * ood_p_align
            
            metrics = evaluate_all(in_scores=id_scores, out_scores=ood_scores)
            
            ood_results['experiments'][f'fg_star_palign_lambda_{lambda_bg}'] = {
                'method': f'fg_star_palign_lambda_{lambda_bg}',
                'metrics': metrics
            }
            
            print(f"  AUROC: {metrics['auroc']:.4f}, FPR95: {metrics['fpr95']:.4f}")
        
        del model_fg
        torch.cuda.empty_cache()
        
        # Store OOD results
        all_results['ood_datasets'][ood_name] = ood_results
        
        # Save intermediate results
        output_dir = Path('./results')
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / 'imagenet_ood_comprehensive_results.json'
        
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Intermediate results saved to: {output_file}")
        print(f"{'='*80}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    for ood_name, ood_data in all_results['ood_datasets'].items():
        print(f"\n{ood_name}:")
        baseline_auroc = ood_data['experiments']['baseline_no_amb']['metrics']['auroc']
        print(f"  Baseline: AUROC = {baseline_auroc:.4f}")
        
        # Best soft+P_align
        best_soft = max(
            [(k, v['metrics']['auroc']) for k, v in ood_data['experiments'].items() if 'soft_palign' in k],
            key=lambda x: x[1]
        )
        print(f"  Best Soft+P_align: {best_soft[0]} = {best_soft[1]:.4f} (Δ={best_soft[1]-baseline_auroc:+.4f})")
        
        # Best fg+P_align
        best_fg = max(
            [(k, v['metrics']['auroc']) for k, v in ood_data['experiments'].items() if 'fg_star_palign' in k],
            key=lambda x: x[1]
        )
        print(f"  Best FG+P_align: {best_fg[0]} = {best_fg[1]:.4f} (Δ={best_fg[1]-baseline_auroc:+.4f})")
    
    print(f"\n{'='*80}")
    print(f"All results saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
