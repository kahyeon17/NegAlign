#!/usr/bin/env python3
"""
Test plain NegLabel on full WaterBirds vs PlacesBG dataset.
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

# Add local utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from clip_negalign import CLIPNegAlign
from ood_evaluate import evaluate_all


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


def main():
    parser = argparse.ArgumentParser(description='Test plain NegLabel on WaterBirds (2-class) vs PlacesBG')
    parser.add_argument('--waterbirds_root', type=str,
                       default='/home/kahyeon/research/bgbias-ood-project/data/waterbirds',
                       help='Path to WaterBirds dataset')
    parser.add_argument('--placesbg_root', type=str,
                       default='/home/kahyeon/research/bgbias-ood-project/data/placesbg',
                       help='Path to PlacesBG dataset')
    parser.add_argument('--output_dir', type=str, default='./results/waterbirds_2class',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='CUDA device')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("Plain NegLabel Evaluation on WaterBirds (2-class) vs PlacesBG")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")

    # Initialize model (plain NegLabel only)
    print("\n" + "="*80)
    print("Initializing plain NegLabel model...")
    print("="*80)

    model = CLIPNegAlign(
        train_dataset='waterbird',
        arch='ViT-B/16',
        seed=0,
        device=args.device,
        output_folder='./data/',
        load_saved_labels=True,  # Use existing negatives
        # Use P_align with plain NegLabel
        use_neglabel_star=False,
        use_role_aware_negatives=True,  # Enable role-aware for star comparison!
        use_p_align=True,
        lambda_bg=1.0,
        pos_topk=10,
        neg_topk=5,
        cam_fg_percentile=80,
        cam_dilate_px=1,
        cam_block=-1
    )

    # Load full datasets
    print("\n" + "="*80)
    print("Loading datasets...")
    print("="*80)

    id_samples = load_waterbirds_samples(args.waterbirds_root, split='test')
    ood_samples = load_placesbg_samples(args.placesbg_root)

    print(f"ID (WaterBirds test, 2-class): {len(id_samples)} samples")
    print(f"OOD (PlacesBG): {len(ood_samples)} samples")

    # Process ID samples
    print("\n" + "="*80)
    print("Processing ID samples (WaterBirds 2-class)...")
    print("="*80)

    id_scores_plain = []  # All negatives (6370)
    id_scores_star = []   # N_obj only (2866)
    id_scores_final = []
    id_p_align = []

    for sample in tqdm(id_samples, desc="ID"):
        img_path = sample['path']
        img = Image.open(img_path).convert('RGB')
        img_tensor = model.clip_preprocess(img).unsqueeze(0).to(model.device)

        # Get scores with P_align
        result = model.detection_score(img_tensor, orig_image=img, return_details=True)
        id_scores_plain.append(result['s_neglabel_plain'])
        id_scores_star.append(result['s_neglabel_star'])
        id_scores_final.append(result['s_final'])
        id_p_align.append(result['p_align'])

    id_scores_plain = np.array(id_scores_plain)
    id_scores_star = np.array(id_scores_star)
    id_scores_final = np.array(id_scores_final)
    id_p_align = np.array(id_p_align)

    print(f"\nID Statistics:")
    print(f"  S_NegLabel_plain (all 6370 negs): mean={id_scores_plain.mean():.4f}, std={id_scores_plain.std():.4f}")
    print(f"  S_NegLabel_star (N_obj 2866):     mean={id_scores_star.mean():.4f}, std={id_scores_star.std():.4f}")
    print(f"  S_final (with P_align): mean={id_scores_final.mean():.4f}, std={id_scores_final.std():.4f}")
    print(f"  P_align: mean={id_p_align.mean():.4f}, std={id_p_align.std():.4f}")

    # Process OOD samples
    print("\n" + "="*80)
    print("Processing OOD samples...")
    print("="*80)

    ood_scores_plain = []
    ood_scores_star = []
    ood_scores_final = []
    ood_p_align = []

    for sample in tqdm(ood_samples, desc="OOD"):
        img_path = sample['path']
        img = Image.open(img_path).convert('RGB')
        img_tensor = model.clip_preprocess(img).unsqueeze(0).to(model.device)

        # Get scores with P_align
        result = model.detection_score(img_tensor, orig_image=img, return_details=True)
        ood_scores_plain.append(result['s_neglabel_plain'])
        ood_scores_star.append(result['s_neglabel_star'])
        ood_scores_final.append(result['s_final'])
        ood_p_align.append(result['p_align'])

    ood_scores_plain = np.array(ood_scores_plain)
    ood_scores_star = np.array(ood_scores_star)
    ood_scores_final = np.array(ood_scores_final)
    ood_p_align = np.array(ood_p_align)

    print(f"\nOOD Statistics:")
    print(f"  S_NegLabel_plain (all 6370 negs): mean={ood_scores_plain.mean():.4f}, std={ood_scores_plain.std():.4f}")
    print(f"  S_NegLabel_star (N_obj 2866):     mean={ood_scores_star.mean():.4f}, std={ood_scores_star.std():.4f}")
    print(f"  S_final (with P_align): mean={ood_scores_final.mean():.4f}, std={ood_scores_final.std():.4f}")
    print(f"  P_align: mean={ood_p_align.mean():.4f}, std={ood_p_align.std():.4f}")

    # Evaluate OOD detection with multiple lambda values
    print("\n" + "="*80)
    print("Evaluating OOD detection with multiple lambda values...")
    print("="*80)

    # Lambda values to test
    lambda_values = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    
    print(f"\nDataset Info:")
    print(f"  ID samples:  {len(id_scores_plain)}")
    print(f"  OOD samples: {len(ood_scores_plain)}")
    
    # ========================================================
    # 1. Plain NegLabel (all 6370 negatives) - baseline
    # ========================================================
    print("\n" + "="*80)
    print("[BASELINE 1] Plain NegLabel (all 6370 negatives)")
    print("="*80)
    auroc_plain, aupr_in_plain, aupr_out_plain, fpr95_plain = evaluate_all(id_scores_plain, ood_scores_plain)
    print(f"  AUROC:    {auroc_plain:.4f}")
    print(f"  FPR95:    {fpr95_plain:.4f}")
    print(f"  AUPR-IN:  {aupr_in_plain:.4f}")
    print(f"  AUPR-OUT: {aupr_out_plain:.4f}")
    
    # ========================================================
    # 2. Star NegLabel (N_obj 2866 only) - baseline
    # ========================================================
    print("\n" + "="*80)
    print("[BASELINE 2] Star NegLabel (N_obj 2866 only)")
    print("="*80)
    auroc_star, aupr_in_star, aupr_out_star, fpr95_star = evaluate_all(id_scores_star, ood_scores_star)
    print(f"  AUROC:    {auroc_star:.4f}")
    print(f"  FPR95:    {fpr95_star:.4f}")
    print(f"  AUPR-IN:  {aupr_in_star:.4f}")
    print(f"  AUPR-OUT: {aupr_out_star:.4f}")
    
    # ========================================================
    # 3. Plain + λ*P_align - lambda search
    # ========================================================
    print("\n" + "="*80)
    print("[METHOD 1] Plain (6370) + λ*P_align")
    print("="*80)
    print(f"{'Lambda':<8} {'AUROC':<8} {'FPR95':<8} {'AUPR-IN':<10} {'AUPR-OUT':<10} {'ID Mean':<10} {'OOD Mean':<10}")
    print("="*80)
    
    lambda_results_plain = []
    for lam in lambda_values:
        id_final = id_scores_plain + lam * id_p_align
        ood_final = ood_scores_plain + lam * ood_p_align
        auroc, aupr_in, aupr_out, fpr95 = evaluate_all(id_final, ood_final)
        
        lambda_results_plain.append({
            'lambda': float(lam),
            'auroc': float(auroc),
            'fpr95': float(fpr95),
            'aupr_in': float(aupr_in),
            'aupr_out': float(aupr_out),
            'id_mean': float(id_final.mean()),
            'ood_mean': float(ood_final.mean())
        })
        print(f"{lam:<8.1f} {auroc:<8.4f} {fpr95:<8.4f} {aupr_in:<10.4f} {aupr_out:<10.4f} {id_final.mean():<10.4f} {ood_final.mean():<10.4f}")
    
    best_idx_plain = np.argmax([r['auroc'] for r in lambda_results_plain])
    best_lambda_plain = lambda_results_plain[best_idx_plain]
    print("="*80)
    print(f"Best Lambda: {best_lambda_plain['lambda']:.1f}, AUROC: {best_lambda_plain['auroc']:.4f}")
    
    # ========================================================
    # 4. Star + λ*P_align - lambda search
    # ========================================================
    print("\n" + "="*80)
    print("[METHOD 2] Star (2866) + λ*P_align")
    print("="*80)
    print(f"{'Lambda':<8} {'AUROC':<8} {'FPR95':<8} {'AUPR-IN':<10} {'AUPR-OUT':<10} {'ID Mean':<10} {'OOD Mean':<10}")
    print("="*80)
    
    lambda_results_star = []
    for lam in lambda_values:
        id_final = id_scores_star + lam * id_p_align
        ood_final = ood_scores_star + lam * ood_p_align
        auroc, aupr_in, aupr_out, fpr95 = evaluate_all(id_final, ood_final)
        
        lambda_results_star.append({
            'lambda': float(lam),
            'auroc': float(auroc),
            'fpr95': float(fpr95),
            'aupr_in': float(aupr_in),
            'aupr_out': float(aupr_out),
            'id_mean': float(id_final.mean()),
            'ood_mean': float(ood_final.mean())
        })
        print(f"{lam:<8.1f} {auroc:<8.4f} {fpr95:<8.4f} {aupr_in:<10.4f} {aupr_out:<10.4f} {id_final.mean():<10.4f} {ood_final.mean():<10.4f}")
    
    best_idx_star = np.argmax([r['auroc'] for r in lambda_results_star])
    best_lambda_star = lambda_results_star[best_idx_star]
    print("="*80)
    print(f"Best Lambda: {best_lambda_star['lambda']:.1f}, AUROC: {best_lambda_star['auroc']:.4f}")

    # Save results
    results = {
        'config': {
            'method': 'comparison_plain_star_palign',
            'dataset': 'waterbirds_2class_vs_placesbg',
            'id_samples': len(id_samples),
            'ood_samples': len(ood_samples),
            'lambda_values': lambda_values,
            'neg_counts': {
                'all_negatives': 6370,
                'n_obj': 2866,
                'n_bg': 2866,
                'n_amb': 638
            }
        },
        'baseline_plain_6370': {
            'method': 'plain NegLabel (all 6370 negatives)',
            'metrics': {
                'auroc': float(auroc_plain),
                'fpr95': float(fpr95_plain),
                'aupr_in': float(aupr_in_plain),
                'aupr_out': float(aupr_out_plain)
            },
            'id_statistics': {
                'mean': float(id_scores_plain.mean()),
                'std': float(id_scores_plain.std()),
                'min': float(id_scores_plain.min()),
                'max': float(id_scores_plain.max()),
                'median': float(np.median(id_scores_plain))
            },
            'ood_statistics': {
                'mean': float(ood_scores_plain.mean()),
                'std': float(ood_scores_plain.std()),
                'min': float(ood_scores_plain.min()),
                'max': float(ood_scores_plain.max()),
                'median': float(np.median(ood_scores_plain))
            }
        },
        'baseline_star_2866': {
            'method': 'star NegLabel (N_obj 2866 only)',
            'metrics': {
                'auroc': float(auroc_star),
                'fpr95': float(fpr95_star),
                'aupr_in': float(aupr_in_star),
                'aupr_out': float(aupr_out_star)
            },
            'id_statistics': {
                'mean': float(id_scores_star.mean()),
                'std': float(id_scores_star.std()),
                'min': float(id_scores_star.min()),
                'max': float(id_scores_star.max()),
                'median': float(np.median(id_scores_star))
            },
            'ood_statistics': {
                'mean': float(ood_scores_star.mean()),
                'std': float(ood_scores_star.std()),
                'min': float(ood_scores_star.min()),
                'max': float(ood_scores_star.max()),
                'median': float(np.median(ood_scores_star))
            }
        },
        'method_plain_palign': {
            'method': 'plain (6370) + λ*P_align',
            'lambda_results': lambda_results_plain,
            'best_result': best_lambda_plain
        },
        'method_star_palign': {
            'method': 'star (2866) + λ*P_align',
            'lambda_results': lambda_results_star,
            'best_result': best_lambda_star
        }
    }

    output_path = os.path.join(args.output_dir, 'comparison_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print(f"Results saved to: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
