#!/usr/bin/env python3
"""
Test NegAlign on WaterBirds dataset.
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


def load_waterbirds_samples(root_dir, split='test', num_samples=None):
    """Load WaterBirds samples (ID: waterbirds)."""
    import pandas as pd

    metadata_path = os.path.join(root_dir, 'metadata.csv')
    metadata = pd.read_csv(metadata_path)

    # Filter for test split
    split_dict = {' train': 0, 'val': 1, 'test': 2}
    metadata = metadata[metadata['split'] == split_dict[split]]

    samples = []
    for idx, row in metadata.iterrows():
        img_path = os.path.join(root_dir, row['img_filename'])
        samples.append({
            'path': img_path,
            'label': row['y']  # 0: landbird, 1: waterbird
        })

        if num_samples and len(samples) >= num_samples:
            break

    return samples


def load_placesbg_samples(root_dir, num_samples=None):
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

            if num_samples and len(samples) >= num_samples:
                break

    return samples


def main():
    parser = argparse.ArgumentParser(description='Test NegAlign on WaterBirds')
    parser.add_argument('--waterbirds_root', type=str, required=True,
                       help='Path to WaterBirds dataset root')
    parser.add_argument('--placesbg_root', type=str, required=True,
                       help='Path to PlacesBG dataset root')
    parser.add_argument('--output_dir', type=str, default='./results/waterbirds_negalign',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='CUDA device')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples per dataset (None = all)')
    # NegAlign args
    parser.add_argument('--use_neglabel_star', action='store_true',
                       help='Use S_NegLabel* modifications')
    parser.add_argument('--use_role_aware', action='store_true',
                       help='Use role-aware negatives (N_obj only)')
    parser.add_argument('--use_p_align', action='store_true',
                       help='Use P_align term')
    parser.add_argument('--neg_split_tau', type=float, default=0.05,
                       help='Threshold for negative split')
    parser.add_argument('--neg_split_dir', type=str, default=None,
                       help='Directory with pre-split negatives (overrides tau)')
    parser.add_argument('--cam_fg_percentile', type=int, default=80,
                       help='CAM foreground percentile')

    args = parser.parse_args()

    # Lambda values to sweep (hardcoded)
    lambda_values = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("NegAlign Evaluation on WaterBirds")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Samples: {args.num_samples or 'all'}")
    print(f"Lambda values: {lambda_values}")
    print(f"Use NegLabel*: {args.use_neglabel_star}")
    print(f"Use role-aware negatives: {args.use_role_aware}")
    print(f"Use P_align: {args.use_p_align}")

    # Initialize model
    print("\n" + "="*80)
    print("Initializing NegAlign model...")
    print("="*80)

    model = CLIPNegAlign(
        train_dataset='imagenet',
        arch='ViT-B/16',
        seed=0,
        device=args.device,
        output_folder='./data/',
        load_saved_labels=True,
        # Negative split
        neg_split_dir=args.neg_split_dir,  # Use custom split dir if provided
        neg_split_tau=args.neg_split_tau,
        neg_use_ambiguous_in_obj=False,
        neg_split_recompute=False,
        # NegLabel* modifications
        use_neglabel_star=args.use_neglabel_star,
        topk_aggregation_k=10,
        use_role_aware_negatives=args.use_role_aware,
        use_scale_stabilization=False,
        # P_align
        use_p_align=args.use_p_align,
        lambda_bg=1.0,  # Will sweep this
        pos_topk=10,
        neg_topk=5,
        # CAM
        cam_fg_percentile=args.cam_fg_percentile,
        cam_dilate_px=1,
        cam_block=-1
    )

    # Load datasets
    print("\n" + "="*80)
    print("Loading datasets...")
    print("="*80)

    id_samples = load_waterbirds_samples(args.waterbirds_root, split='test', num_samples=args.num_samples)
    ood_samples = load_placesbg_samples(args.placesbg_root, num_samples=args.num_samples)

    print(f"ID (WaterBirds): {len(id_samples)} samples")
    print(f"OOD (PlacesBG): {len(ood_samples)} samples")

    # Process ID samples
    print("\n" + "="*80)
    print("Processing ID samples...")
    print("="*80)

    id_scores_plain = []
    id_scores_star = []
    id_p_align = []
    id_cam_valid = []

    for sample in tqdm(id_samples, desc="ID"):
        img = Image.open(sample['path']).convert('RGB')
        img_tensor = model.clip_preprocess(img).unsqueeze(0).to(args.device)

        details = model.detection_score(img_tensor, orig_image=img, return_details=True)

        id_scores_plain.append(details['s_neglabel_plain'])
        id_scores_star.append(details['s_neglabel_star'])
        id_p_align.append(details['p_align'])
        id_cam_valid.append(details['cam_valid'])

    id_scores_plain = np.array(id_scores_plain)
    id_scores_star = np.array(id_scores_star)
    id_p_align = np.array(id_p_align)
    id_cam_valid = np.array(id_cam_valid)

    print(f"\nID Statistics:")
    print(f"  S_NegLabel_plain: mean={id_scores_plain.mean():.4f}, std={id_scores_plain.std():.4f}")
    print(f"  S_NegLabel*:      mean={id_scores_star.mean():.4f}, std={id_scores_star.std():.4f}")
    print(f"  P_align:          mean={id_p_align.mean():.4f}, std={id_p_align.std():.4f}")
    print(f"  CAM valid rate:   {id_cam_valid.sum() / len(id_cam_valid):.2%}")

    # Process OOD samples
    print("\n" + "="*80)
    print("Processing OOD samples...")
    print("="*80)

    ood_scores_plain = []
    ood_scores_star = []
    ood_p_align = []
    ood_cam_valid = []

    for sample in tqdm(ood_samples, desc="OOD"):
        img = Image.open(sample['path']).convert('RGB')
        img_tensor = model.clip_preprocess(img).unsqueeze(0).to(args.device)

        details = model.detection_score(img_tensor, orig_image=img, return_details=True)

        ood_scores_plain.append(details['s_neglabel_plain'])
        ood_scores_star.append(details['s_neglabel_star'])
        ood_p_align.append(details['p_align'])
        ood_cam_valid.append(details['cam_valid'])

    ood_scores_plain = np.array(ood_scores_plain)
    ood_scores_star = np.array(ood_scores_star)
    ood_p_align = np.array(ood_p_align)
    ood_cam_valid = np.array(ood_cam_valid)

    print(f"\nOOD Statistics:")
    print(f"  S_NegLabel_plain: mean={ood_scores_plain.mean():.4f}, std={ood_scores_plain.std():.4f}")
    print(f"  S_NegLabel*:      mean={ood_scores_star.mean():.4f}, std={ood_scores_star.std():.4f}")
    print(f"  P_align:          mean={ood_p_align.mean():.4f}, std={ood_p_align.std():.4f}")
    print(f"  CAM valid rate:   {ood_cam_valid.sum() / len(ood_cam_valid):.2%}")

    # Evaluate across lambda values
    print("\n" + "="*80)
    print("Evaluating OOD detection...")
    print("="*80)

    # First evaluate plain NegLabel score (baseline)
    print("\n[Baseline] NegLabel Plain (all negatives):")
    print("-" * 40)
    # Print score distributions
    print(f"  ID scores:  min={id_scores_plain.min():.4f}, max={id_scores_plain.max():.4f}, median={np.median(id_scores_plain):.4f}")
    print(f"  OOD scores: min={ood_scores_plain.min():.4f}, max={ood_scores_plain.max():.4f}, median={np.median(ood_scores_plain):.4f}")
    try:
        auroc_plain, aupr_in_plain, aupr_out_plain, fpr95_plain = evaluate_all(id_scores_plain, ood_scores_plain)
        metrics_plain = {
            'auroc': float(auroc_plain),
            'aupr_in': float(aupr_in_plain),
            'aupr_out': float(aupr_out_plain),
            'fpr95': float(fpr95_plain)
        }
        print(f"  AUROC:    {metrics_plain['auroc']:.4f}")
        print(f"  FPR95:    {metrics_plain['fpr95']:.4f}")
        print(f"  AUPR-IN:  {metrics_plain['aupr_in']:.4f}")
        print(f"  AUPR-OUT: {metrics_plain['aupr_out']:.4f}")
    except Exception as e:
        print(f"  Error in plain evaluation: {e}")
        metrics_plain = {'auroc': 0.0, 'aupr_in': 0.0, 'aupr_out': 0.0, 'fpr95': 1.0}

    # Evaluate with NegLabel* (if enabled)
    if args.use_neglabel_star:
        print("\n[With NegLabel*] S_NegLabel* only (λ=0):")
        print("-" * 40)
        try:
            auroc_star, aupr_in_star, aupr_out_star, fpr95_star = evaluate_all(id_scores_star, ood_scores_star)
            metrics_star = {
                'auroc': float(auroc_star),
                'aupr_in': float(aupr_in_star),
                'aupr_out': float(aupr_out_star),
                'fpr95': float(fpr95_star)
            }
            print(f"  AUROC:    {metrics_star['auroc']:.4f}")
            print(f"  FPR95:    {metrics_star['fpr95']:.4f}")
            print(f"  AUPR-IN:  {metrics_star['aupr_in']:.4f}")
            print(f"  AUPR-OUT: {metrics_star['aupr_out']:.4f}")
        except Exception as e:
            print(f"  Error in star evaluation: {e}")
            metrics_star = {'auroc': 0.0, 'aupr_in': 0.0, 'aupr_out': 0.0, 'fpr95': 1.0}
    else:
        metrics_star = metrics_plain

    # Lambda sweep for NegAlign (S_NegLabel* + λ * P_align)
    print("\n[NegAlign] Lambda Sweep (S_NegLabel* + λ * P_align):")
    print("-" * 80)

    lambda_sweep = {}
    best_auroc = -1
    best_lambda = None

    for lam in lambda_values:
        # Compute final scores
        id_final = id_scores_star + lam * id_p_align
        ood_final = ood_scores_star + lam * ood_p_align

        # Evaluate
        try:
            auroc, aupr_in, aupr_out, fpr95 = evaluate_all(id_final, ood_final)
            
            metrics = {
                'auroc': float(auroc),
                'aupr_in': float(aupr_in),
                'aupr_out': float(aupr_out),
                'fpr95': float(fpr95)
            }
        except Exception as e:
            print(f"  Error at λ={lam}: {e}")
            metrics = {
                'auroc': 0.0,
                'aupr_in': 0.0,
                'aupr_out': 0.0,
                'fpr95': 1.0
            }

        lambda_sweep[str(lam)] = {
            'lambda': lam,
            'metrics': metrics
        }

        print(f"  λ = {lam:5.1f}:  AUROC: {metrics['auroc']:.4f}  |  FPR95: {metrics['fpr95']:.4f}  |  AUPR-IN: {metrics['aupr_in']:.4f}  |  AUPR-OUT: {metrics['aupr_out']:.4f}")

        if metrics['auroc'] > best_auroc:
            best_auroc = metrics['auroc']
            best_lambda = lam

    print(f"\n{'='*80}")
    print(f"Best Lambda: {best_lambda} (AUROC: {best_auroc:.4f})")
    print(f"{'='*80}")

    # Save results
    results_summary = {
        'config': {
            'use_neglabel_star': args.use_neglabel_star,
            'use_role_aware': args.use_role_aware,
            'use_p_align': args.use_p_align,
            'neg_split_tau': args.neg_split_tau,
            'cam_fg_percentile': args.cam_fg_percentile,
        },
        'baseline': {
            'method': 'NegLabel_plain',
            'metrics': metrics_plain
        },
        'neglabel_star': {
            'method': 'NegLabel*',
            'metrics': metrics_star
        },
        'best_lambda': best_lambda,
        'best_auroc': best_auroc,
        'lambda_sweep': lambda_sweep,
        'statistics': {
            'id': {
                's_neglabel_plain_mean': float(id_scores_plain.mean()),
                's_neglabel_plain_std': float(id_scores_plain.std()),
                's_neglabel_star_mean': float(id_scores_star.mean()),
                's_neglabel_star_std': float(id_scores_star.std()),
                'p_align_mean': float(id_p_align.mean()),
                'p_align_std': float(id_p_align.std()),
                'cam_valid_rate': float(id_cam_valid.sum() / len(id_cam_valid))
            },
            'ood': {
                's_neglabel_plain_mean': float(ood_scores_plain.mean()),
                's_neglabel_plain_std': float(ood_scores_plain.std()),
                's_neglabel_star_mean': float(ood_scores_star.mean()),
                's_neglabel_star_std': float(ood_scores_star.std()),
                'p_align_mean': float(ood_p_align.mean()),
                'p_align_std': float(ood_p_align.std()),
                'cam_valid_rate': float(ood_cam_valid.sum() / len(ood_cam_valid))
            }
        }
    }

    output_file = os.path.join(args.output_dir, 'results.json')
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print("="*80)


if __name__ == '__main__':
    main()
