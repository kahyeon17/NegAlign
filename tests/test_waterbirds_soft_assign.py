#!/usr/bin/env python3
"""
Test soft assignment of ambiguous negatives on WaterBirds dataset.

Compares:
1. Baseline (use_amb_soft=False, lambda=0)
2. Soft assignment (use_amb_soft=True, lambda=0)
3. Soft assignment + P_align (use_amb_soft=True, lambda sweep)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
from datetime import datetime
from pathlib import Path

from core.clip_negalign import CLIPNegAlign
from core.neglabel_soft_assign import CLIPNegAlignSoftAssign
from core.neglabel_foreground_star import CLIPNegAlignForegroundStar
from utils.ood_evaluate import evaluate_all
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


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


def load_waterbirds_samples(root_dir, split='test'):
    """Load WaterBirds samples (ID: 2-class waterbird/landbird)."""
    import pandas as pd

    metadata_path = os.path.join(root_dir, 'metadata.csv')
    metadata = pd.read_csv(metadata_path)

    # Filter for test split
    split_dict = {'train': 0, 'val': 1, 'test': 2}
    metadata = metadata[metadata['split'] == split_dict[split]]

    samples = []
    for idx, row in metadata.iterrows():
        img_path = os.path.join(root_dir, row['img_filename'])
        samples.append({
            'path': img_path,
            'label': row['y']  # 0: landbird, 1: waterbird
        })

    return samples


def load_placesbg_samples(root_dir):
    """Load PlacesBG samples (OOD: background-only scenes)."""
    # Try common directory names
    for subdir in ['background', 'images', 'val']:
        image_dir = os.path.join(root_dir, subdir)
        if os.path.exists(image_dir):
            break
    else:
        image_dir = root_dir

    samples = []
    for img_file in sorted(os.listdir(image_dir)):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_dir, img_file)
            samples.append({
                'path': img_path,
                'label': -1  # OOD
            })

    return samples


def load_waterbirds_test():
    """Load WaterBirds test set (5,794 samples)."""
    waterbirds_root = '/home/kahyeon/research/bgbias-ood-project/data/waterbirds'
    return load_waterbirds_samples(waterbirds_root, split='test')


def load_placesbg_ood():
    """Load PlacesBG as OOD (10,000 samples)."""
    placesbg_root = '/home/kahyeon/research/bgbias-ood-project/data/placesbg'
    return load_placesbg_samples(placesbg_root)


def evaluate_method(model, id_samples, ood_samples, method_name):
    """Evaluate a single method configuration."""
    from PIL import Image
    import numpy as np
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {method_name}")
    print(f"{'='*80}")
    
    id_scores = []
    ood_scores = []
    
    # Process ID samples
    print(f"Processing {len(id_samples)} ID samples...")
    for idx, sample in enumerate(id_samples):
        try:
            img_pil = Image.open(sample['path']).convert('RGB')
            img = model.clip_preprocess(img_pil).unsqueeze(0).to(model.device)
            
            with torch.no_grad():
                score = model.detection_score(img, orig_image=img_pil)
            
            id_scores.append(float(score))
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(id_samples)} ID samples")
        
        except Exception as e:
            print(f"Error processing ID sample {idx}: {e}")
            continue
    
    # Process OOD samples
    print(f"\nProcessing {len(ood_samples)} OOD samples...")
    for idx, sample in enumerate(ood_samples):
        try:
            img_pil = Image.open(sample['path']).convert('RGB')
            img = model.clip_preprocess(img_pil).unsqueeze(0).to(model.device)
            
            with torch.no_grad():
                score = model.detection_score(img, orig_image=img_pil)
            
            ood_scores.append(float(score))
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(ood_samples)} OOD samples")
        
        except Exception as e:
            print(f"Error processing OOD sample {idx}: {e}")
            continue
    
    # Compute metrics
    print(f"\nComputing metrics...")
    print(f"  ID scores: {len(id_scores)}")
    print(f"  OOD scores: {len(ood_scores)}")
    
    # Convert to numpy arrays
    id_scores = np.array(id_scores)
    ood_scores = np.array(ood_scores)
    
    # Print score statistics
    print(f"  ID score range: [{id_scores.min():.4f}, {id_scores.max():.4f}], mean={id_scores.mean():.4f}")
    print(f"  OOD score range: [{ood_scores.min():.4f}, {ood_scores.max():.4f}], mean={ood_scores.mean():.4f}")
    
    metrics = evaluate_all(id_scores, ood_scores)
    
    print(f"\nResults for {method_name}:")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    print(f"  FPR95: {metrics['fpr95']:.4f}")
    print(f"  AUPR-IN: {metrics['aupr_in']:.4f}")
    print(f"  AUPR-OUT: {metrics['aupr_out']:.4f}")
    
    return {
        'method': method_name,
        'id_scores': id_scores,
        'ood_scores': ood_scores,
        'metrics': metrics
    }


def main():
    print("="*80)
    print("WaterBirds Soft Assignment Test")
    print("="*80)
    
    # Load datasets
    print("\nLoading datasets...")
    id_samples = load_waterbirds_test()
    ood_samples = load_placesbg_ood()
    print(f"  ID (WaterBirds test): {len(id_samples)} samples")
    print(f"  OOD (PlacesBG): {len(ood_samples)} samples")
    
    # Results storage
    results = {
        'timestamp': datetime.now().isoformat(),
        'id_dataset': 'WaterBirds test',
        'ood_dataset': 'PlacesBG',
        'id_samples': len(id_samples),
        'ood_samples': len(ood_samples),
        'experiments': {}
    }
    
    # ========================================================================
    # Baseline: No soft assignment, no P_align
    # ========================================================================
    print("\n" + "="*80)
    print("1. Baseline: use_amb_soft=False, lambda=0")
    print("="*80)
    
    model_baseline = CLIPNegAlign(
        train_dataset='waterbird',
        arch='ViT-B/16',
        seed=0,
        device='cuda:0',
        output_folder='./data/',
        load_saved_labels=True,
        use_neglabel_star=False,
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
    results['experiments']['baseline_no_amb'] = result_baseline
    
    # ========================================================================
    # Soft assignment only: use_amb_soft=True, lambda=0
    # ========================================================================
    print("\n" + "="*80)
    print("2. Soft Assignment Only: use_amb_soft=True, lambda=0")
    print("="*80)
    
    model_soft = CLIPNegAlignSoftAssign(
        train_dataset='waterbird',
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
    results['experiments']['soft_only'] = result_soft
    
    # ========================================================================
    # Soft assignment + P_align: OPTIMIZED Lambda sweep
    # ========================================================================
    print("\n" + "="*80)
    print("3. Soft Assignment + P_align: OPTIMIZED Lambda sweep")
    print("="*80)
    print("(Pre-computes scores once, then adjusts lambda - 8x faster!)\n")
    
    # Initialize model with P_align enabled
    print("Initializing soft assignment model with P_align...")
    model_soft_palign = CLIPNegAlignSoftAssign(
        train_dataset='waterbird',
        arch='ViT-B/16',
        seed=0,
        device='cuda:0',
        output_folder='./data/',
        load_saved_labels=True,
        use_neglabel_star=True,
        use_role_aware_negatives=True,
        use_p_align=True,
        lambda_bg=1.0,  # Will be adjusted
        use_amb_soft=True,
        amb_soft_beta=10.0
    )
    
    # Pre-compute S_NegLabel_star and P_align for ALL images ONCE
    print("\nPre-computing scores (ONE-TIME)...")
    print("Processing ID samples...")
    id_s_neglabel_soft = []
    id_p_align_soft = []
    for sample in id_samples:
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
    for sample in ood_samples:
        img, _ = load_image(sample['path'])
        img = img.to(model_soft_palign.device)
        details = model_soft_palign.detection_score(img.unsqueeze(0), orig_image=None, return_details=True)
        ood_s_neglabel_soft.append(details['s_neglabel_star'])
        ood_p_align_soft.append(details['p_align'])
    
    ood_s_neglabel_soft = np.array(ood_s_neglabel_soft)
    ood_p_align_soft = np.array(ood_p_align_soft)
    print(f"  S_NegLabel_star range: [{ood_s_neglabel_soft.min():.4f}, {ood_s_neglabel_soft.max():.4f}]")
    print(f"  P_align range: [{ood_p_align_soft.min():.4f}, {ood_p_align_soft.max():.4f}]")
    
    # Lambda sweep: just compute S_final = S_NegLabel + lambda * P_align (instant!)
    print("\nLambda sweep (OPTIMIZED - no image reprocessing)...")
    lambda_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    
    for lambda_bg in lambda_values:
        print(f"\n--- Lambda = {lambda_bg} ---")
        
        # Compute final scores (instant!)
        id_scores = id_s_neglabel_soft + lambda_bg * id_p_align_soft
        ood_scores = ood_s_neglabel_soft + lambda_bg * ood_p_align_soft
        
        # Evaluate
        metrics = evaluate_all(
            in_scores=id_scores,
            out_scores=ood_scores
        )
        
        results['experiments'][f'soft_palign_lambda_{lambda_bg}'] = {
            'method': f'soft_palign_lambda_{lambda_bg}',
            'metrics': metrics
        }
        
        print(f"Results for soft_palign_lambda_{lambda_bg}:")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  FPR95: {metrics['fpr95']:.4f}")
        print(f"  AUPR-IN: {metrics['aupr_in']:.4f}")
        print(f"  AUPR-OUT: {metrics['aupr_out']:.4f}")
    
    # Clean up model
    del model_soft_palign
    torch.cuda.empty_cache()
    
    # ========================================================================
    # Foreground-based Star + P_align: OPTIMIZED Lambda sweep
    # ========================================================================
    print("\n" + "="*80)
    print("4. Foreground-based Star + P_align: OPTIMIZED Lambda sweep")
    print("="*80)
    print("(Pre-computes scores once, then adjusts lambda - 8x faster!)\n")
    
    # Initialize model with P_align enabled
    print("Initializing foreground-based model...")
    model_fg = CLIPNegAlignForegroundStar(
        train_dataset='waterbird',
        arch='ViT-B/16',
        seed=0,
        device='cuda:0',
        output_folder='./data/',
        load_saved_labels=True,
        use_role_aware_negatives=True,
        use_p_align=True,
        lambda_bg=1.0,  # Will be adjusted
        cam_fg_percentile=80,
        cam_dilate_px=1,
        cam_block=-1
    )
    
    # Pre-compute S_NegLabel_star and P_align for ALL images ONCE
    print("\nPre-computing scores (ONE-TIME)...")
    print("Processing ID samples...")
    id_s_neglabel = []
    id_p_align = []
    for sample in id_samples:
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
    for sample in ood_samples:
        img, _ = load_image(sample['path'])
        img = img.to(model_fg.device)
        details = model_fg.detection_score(img.unsqueeze(0), orig_image=None, return_details=True)
        ood_s_neglabel.append(details['s_neglabel_star'])
        ood_p_align.append(details['p_align'])
    
    ood_s_neglabel = np.array(ood_s_neglabel)
    ood_p_align = np.array(ood_p_align)
    print(f"  S_NegLabel_star range: [{ood_s_neglabel.min():.4f}, {ood_s_neglabel.max():.4f}]")
    print(f"  P_align range: [{ood_p_align.min():.4f}, {ood_p_align.max():.4f}]")
    
    # Lambda sweep: just compute S_final = S_NegLabel + lambda * P_align (instant!)
    print("\nLambda sweep (OPTIMIZED - no image reprocessing)...")
    lambda_values_fg = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    
    for lambda_bg in lambda_values_fg:
        print(f"\n--- Lambda = {lambda_bg} ---")
        
        # Compute final scores (instant!)
        id_scores = id_s_neglabel + lambda_bg * id_p_align
        ood_scores = ood_s_neglabel + lambda_bg * ood_p_align
        
        # Evaluate
        metrics = evaluate_all(
            in_scores=id_scores,
            out_scores=ood_scores
        )
        
        exp_name = f'fg_star_palign_lambda_{lambda_bg}'
        results['experiments'][exp_name] = {
            'method': f'foreground_star_palign_lambda_{lambda_bg}',
            'metrics': metrics
        }
        
        print(f"Results for {exp_name}:")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  FPR95: {metrics['fpr95']:.4f}")
        print(f"  AUPR-IN: {metrics['aupr_in']:.4f}")
        print(f"  AUPR-OUT: {metrics['aupr_out']:.4f}")
    
    # Clean up
    del model_fg
    torch.cuda.empty_cache()
    
    # ========================================================================
    # Save results
    # ========================================================================
    output_dir = Path('./results')
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'waterbirds_soft_assign_results.json'
    
    # Convert to serializable format
    results_serializable = {
        'timestamp': results['timestamp'],
        'id_dataset': results['id_dataset'],
        'ood_dataset': results['ood_dataset'],
        'id_samples': results['id_samples'],
        'ood_samples': results['ood_samples'],
        'experiments': {}
    }
    
    for exp_name, exp_data in results['experiments'].items():
        results_serializable['experiments'][exp_name] = {
            'method': exp_data['method'],
            'metrics': exp_data['metrics']
        }
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # ========================================================================
    # Summary comparison
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY: Soft Assignment & Foreground-based Results")
    print("="*80)
    
    print("\n1. Baseline vs Soft Assignment (no P_align):")
    baseline_auroc = results['experiments']['baseline_no_amb']['metrics']['auroc']
    soft_auroc = results['experiments']['soft_only']['metrics']['auroc']
    print(f"  Baseline (no amb):  AUROC = {baseline_auroc:.4f}")
    print(f"  Soft assignment:    AUROC = {soft_auroc:.4f}")
    print(f"  Improvement:        {soft_auroc - baseline_auroc:+.4f}")
    
    print("\n2. Soft Assignment + P_align (lambda sweep):")
    best_soft_lambda = None
    best_soft_auroc = soft_auroc
    
    for lambda_bg in lambda_values:
        exp_name = f'soft_palign_lambda_{lambda_bg}'
        auroc = results['experiments'][exp_name]['metrics']['auroc']
        improvement = auroc - soft_auroc
        print(f"  λ={lambda_bg}:  AUROC = {auroc:.4f}  (Δ={improvement:+.4f})")
        
        if auroc > best_soft_auroc:
            best_soft_auroc = auroc
            best_soft_lambda = lambda_bg
    
    print("\n3. Foreground-based Star + P_align (lambda sweep):")
    best_fg_lambda = None
    best_fg_auroc = baseline_auroc
    
    for lambda_bg in lambda_values_fg:
        exp_name = f'fg_star_palign_lambda_{lambda_bg}'
        auroc = results['experiments'][exp_name]['metrics']['auroc']
        improvement = auroc - baseline_auroc
        print(f"  λ={lambda_bg}:  AUROC = {auroc:.4f}  (Δ={improvement:+.4f})")
        
        if auroc > best_fg_auroc:
            best_fg_auroc = auroc
            best_fg_lambda = lambda_bg
    
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS:")
    print("="*80)
    
    print(f"\n1. Soft assignment + P_align:")
    if best_soft_lambda is not None:
        print(f"   λ={best_soft_lambda}:  AUROC = {best_soft_auroc:.4f}  (Δ={best_soft_auroc - baseline_auroc:+.4f})")
    else:
        print(f"   No improvement (best is soft-only: {soft_auroc:.4f})")
    
    print(f"\n2. Foreground-based star + P_align:")
    if best_fg_lambda is not None:
        print(f"   λ={best_fg_lambda}:  AUROC = {best_fg_auroc:.4f}  (Δ={best_fg_auroc - baseline_auroc:+.4f})")
    else:
        print(f"   No improvement")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
