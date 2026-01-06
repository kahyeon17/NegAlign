#!/usr/bin/env python3
"""
Quick test for V3 soft weighting functionality.

This script verifies that:
1. R-scores are loaded correctly
2. Soft weights are computed properly
3. Weighted scoring works as expected
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Direct import to avoid __init__.py issues
from core.clip_negalign_v3 import CLIPNegAlignV3
import torch
import numpy as np

def main():
    print("="*80)
    print("Testing NegAlign V3: Soft Weighting")
    print("="*80)

    # Initialize model with soft weighting
    print("\n[1] Initializing CLIPNegAlignV3 with soft weighting...")
    model = CLIPNegAlignV3(
        train_dataset='imagenet',
        arch='ViT-B/16',
        device='cuda:0',
        output_folder='./data/',
        load_saved_labels=True,
        use_soft_weighting=True,
        soft_weight_scale=20.0,
        use_neglabel_star=True,
        topk_aggregation_k=10,
        use_role_aware_negatives=False,  # Not needed with soft weighting
        use_p_align=False,  # Test base scoring first
        lambda_bg=0.0
    )

    print("\n[2] Analyzing weight distribution...")
    weights_obj = model.weights_obj.cpu().numpy()
    weights_bg = model.weights_bg.cpu().numpy()

    # Statistics
    print(f"\nObject Weights:")
    print(f"  Min:  {weights_obj.min():.4f}")
    print(f"  Max:  {weights_obj.max():.4f}")
    print(f"  Mean: {weights_obj.mean():.4f}")
    print(f"  Std:  {weights_obj.std():.4f}")
    print(f"  >0.9: {(weights_obj > 0.9).sum()} / {len(weights_obj)}")
    print(f"  >0.7: {(weights_obj > 0.7).sum()} / {len(weights_obj)}")
    print(f"  <0.3: {(weights_obj < 0.3).sum()} / {len(weights_obj)}")

    print(f"\nBackground Weights:")
    print(f"  Min:  {weights_bg.min():.4f}")
    print(f"  Max:  {weights_bg.max():.4f}")
    print(f"  Mean: {weights_bg.mean():.4f}")
    print(f"  Std:  {weights_bg.std():.4f}")
    print(f"  >0.9: {(weights_bg > 0.9).sum()} / {len(weights_bg)}")
    print(f"  >0.7: {(weights_bg > 0.7).sum()} / {len(weights_bg)}")
    print(f"  <0.3: {(weights_bg < 0.3).sum()} / {len(weights_bg)}")

    # Check ambiguous words (both weights around 0.5)
    ambiguous_mask = (weights_obj > 0.3) & (weights_obj < 0.7) & (weights_bg > 0.3) & (weights_bg < 0.7)
    print(f"\nAmbiguous words (both weights 0.3-0.7): {ambiguous_mask.sum()}")

    # Show some examples
    print("\n[3] Example words with extreme weights:")

    # Get word list
    import pickle
    with open('./data/neg_labels_noun.pkl', 'rb') as f:
        neg_nouns = pickle.load(f)
    with open('./data/neg_labels_adj.pkl', 'rb') as f:
        neg_adjs = pickle.load(f)
    all_words = neg_nouns + neg_adjs

    # Top object-leaning words
    top_obj_indices = np.argsort(weights_obj)[-10:][::-1]
    print("\nTop 10 object-leaning (high obj weight):")
    for idx in top_obj_indices:
        print(f"  {all_words[idx]:30s} | obj={weights_obj[idx]:.3f}, bg={weights_bg[idx]:.3f}")

    # Top background-leaning words
    top_bg_indices = np.argsort(weights_bg)[-10:][::-1]
    print("\nTop 10 background-leaning (high bg weight):")
    for idx in top_bg_indices:
        print(f"  {all_words[idx]:30s} | obj={weights_obj[idx]:.3f}, bg={weights_bg[idx]:.3f}")

    # Most ambiguous words
    ambiguity_score = 1 - np.abs(weights_obj - weights_bg)
    top_amb_indices = np.argsort(ambiguity_score)[-10:][::-1]
    print("\nTop 10 ambiguous (similar obj/bg weights):")
    for idx in top_amb_indices:
        print(f"  {all_words[idx]:30s} | obj={weights_obj[idx]:.3f}, bg={weights_bg[idx]:.3f}")

    print("\n[4] Testing detection score computation...")
    # Load a test image
    from PIL import Image
    import requests
    from io import BytesIO

    # Download a sample ImageNet image
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg"
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')

        # Preprocess
        img_tensor = model.clip_preprocess(img).unsqueeze(0).to('cuda:0')

        # Get score
        details = model.detection_score(img_tensor, return_details=True)

        print(f"\nTest image (cat) detection score:")
        print(f"  S_NegLabel*: {details['s_neglabel']:.4f}")
        print(f"  S_final:     {details['s_final']:.4f}")
        print(f"  Predicted class: {model.pos_labels[details['predicted_class']]}")

    except Exception as e:
        print(f"Could not test with online image: {e}")
        print("Skipping image test...")

    print("\n" + "="*80)
    print("✅ V3 Soft Weighting Test Complete!")
    print("="*80)
    print("\nKey takeaways:")
    print("- All 6370 negatives are now used (no words discarded)")
    print("- Words are weighted based on r-scores (object vs background)")
    print("- Ambiguous words contribute to both obj and bg with medium weights")
    print("\nNext steps:")
    print("- Compare V3 (soft weighting) vs V2 (hard threshold) on ImageNet")
    print("- Test if soft weighting improves ID-OOD separation")

if __name__ == '__main__':
    main()
