#!/usr/bin/env python3
"""
Example usage of NegAlign for single-image OOD detection.
"""

import sys
import os
from PIL import Image

# Add local utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from clip_negalign import CLIPNegAlign, sanity_check


def example_basic():
    """Example 1: Basic usage with default settings."""
    print("\n" + "="*80)
    print("Example 1: Basic Usage")
    print("="*80)

    # Initialize model
    model = CLIPNegAlign(
        train_dataset='imagenet',
        device='cuda:0',
        output_folder='./data/',
        load_saved_labels=True,
        use_p_align=False  # Baseline: S_NegLabel only
    )

    # Process image
    img = Image.new('RGB', (224, 224), color='blue')
    img_tensor = model.clip_preprocess(img).unsqueeze(0).to('cuda:0')

    # Get score
    score = model.detection_score(img_tensor)
    print(f"OOD score (baseline): {score:.4f}")


def example_with_p_align():
    """Example 2: Using P_align for background-aware detection."""
    print("\n" + "="*80)
    print("Example 2: With P_align")
    print("="*80)

    model = CLIPNegAlign(
        train_dataset='imagenet',
        device='cuda:0',
        output_folder='./data/',
        load_saved_labels=True,

        # Enable P_align
        use_p_align=True,
        lambda_bg=2.0,
        cam_fg_percentile=80
    )

    # Process image
    img = Image.new('RGB', (224, 224), color='green')
    img_tensor = model.clip_preprocess(img).unsqueeze(0).to('cuda:0')

    # Get detailed breakdown
    details = model.detection_score(img_tensor, orig_image=img, return_details=True)

    print(f"S_NegLabel:  {details['s_neglabel_plain']:.4f}")
    print(f"P_align:     {details['p_align']:.4f}")
    print(f"S_final:     {details['s_final']:.4f}")
    print(f"CAM valid:   {details['cam_valid']}")


def example_full_negalign():
    """Example 3: Full NegAlign with all modifications."""
    print("\n" + "="*80)
    print("Example 3: Full NegAlign")
    print("="*80)

    model = CLIPNegAlign(
        train_dataset='imagenet',
        device='cuda:0',
        output_folder='./data/',
        load_saved_labels=True,

        # Enable all modifications
        use_neglabel_star=True,
        use_role_aware_negatives=True,
        use_scale_stabilization=False,

        # Enable P_align
        use_p_align=True,
        lambda_bg=2.0,

        # CAM settings
        cam_fg_percentile=80,
        cam_dilate_px=1,
        cam_block=-1
    )

    # Run sanity check
    sanity_check(model)


def example_real_image(image_path):
    """Example 4: Process real image."""
    print("\n" + "="*80)
    print("Example 4: Real Image")
    print("="*80)

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return

    model = CLIPNegAlign(
        train_dataset='imagenet',
        device='cuda:0',
        output_folder='./data/',
        load_saved_labels=True,
        use_neglabel_star=True,
        use_role_aware_negatives=True,
        use_p_align=True,
        lambda_bg=2.0
    )

    # Load and process image
    img = Image.open(image_path).convert('RGB')
    img_tensor = model.clip_preprocess(img).unsqueeze(0).to('cuda:0')

    # Get details
    details = model.detection_score(img_tensor, orig_image=img, return_details=True)

    print(f"Image: {image_path}")
    print(f"Predicted class: {model.pos_labels[details['predicted_class_idx']]}")
    print(f"S_NegLabel*:     {details['s_neglabel_star']:.4f}")
    print(f"P_align:         {details['p_align']:.4f}")
    print(f"S_final:         {details['s_final']:.4f}")
    print(f"CAM valid:       {details['cam_valid']}")

    # Interpretation
    if details['s_final'] > 0.5:
        print("\n-> Likely IN-distribution")
    else:
        print("\n-> Likely OUT-of-distribution")


def example_ablation():
    """Example 5: Ablation study (compare different configs)."""
    print("\n" + "="*80)
    print("Example 5: Ablation Study")
    print("="*80)

    # Test image
    img = Image.new('RGB', (224, 224), color='red')

    configs = [
        {
            'name': 'Baseline (S_NegLabel_plain)',
            'use_neglabel_star': False,
            'use_role_aware_negatives': False,
            'use_p_align': False
        },
        {
            'name': 'S_NegLabel* (role-aware)',
            'use_neglabel_star': True,
            'use_role_aware_negatives': True,
            'use_p_align': False
        },
        {
            'name': 'P_align only',
            'use_neglabel_star': False,
            'use_role_aware_negatives': False,
            'use_p_align': True,
            'lambda_bg': 2.0
        },
        {
            'name': 'Full NegAlign',
            'use_neglabel_star': True,
            'use_role_aware_negatives': True,
            'use_p_align': True,
            'lambda_bg': 2.0
        }
    ]

    for config in configs:
        name = config.pop('name')

        model = CLIPNegAlign(
            train_dataset='imagenet',
            device='cuda:0',
            output_folder='./data/',
            load_saved_labels=True,
            **config
        )

        img_tensor = model.clip_preprocess(img).unsqueeze(0).to('cuda:0')
        score = model.detection_score(img_tensor, orig_image=img)

        print(f"{name:35s}: {score:.4f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NegAlign usage examples')
    parser.add_argument('--example', type=int, default=0,
                       help='Example number (0=all, 1-5=specific)')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image for example 4')

    args = parser.parse_args()

    if args.example == 0 or args.example == 1:
        example_basic()

    if args.example == 0 or args.example == 2:
        example_with_p_align()

    if args.example == 0 or args.example == 3:
        example_full_negalign()

    if args.example == 4:
        if args.image:
            example_real_image(args.image)
        else:
            print("Error: --image required for example 4")

    if args.example == 0 or args.example == 5:
        example_ablation()

    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)
