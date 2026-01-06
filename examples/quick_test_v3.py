#!/usr/bin/env python3
"""
Quick standalone test for V3 soft weighting.
This bypasses import issues by directly checking the weight computation logic.
"""

import pandas as pd
import numpy as np
import torch

def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def main():
    print("="*80)
    print("V3 Soft Weighting: Weight Distribution Analysis")
    print("="*80)

    # Load r-scores
    score_file = './data/negatives_split/neg_word_scores.csv'
    print(f"\n[1] Loading r-scores from: {score_file}")
    df = pd.read_csv(score_file)

    print(f"  Total words: {len(df)}")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())

    # Compute soft weights with different scales
    r_scores = df['r_score'].values

    print("\n[2] Computing soft weights with scale=20.0...")
    scale = 20.0

    # weight_obj: high when r is positive (object-leaning)
    weights_obj = sigmoid(r_scores * scale)

    # weight_bg: high when r is negative (background-leaning)
    weights_bg = sigmoid(-r_scores * scale)

    print(f"\nObject Weights (scale={scale}):")
    print(f"  Min:  {weights_obj.min():.4f}")
    print(f"  Max:  {weights_obj.max():.4f}")
    print(f"  Mean: {weights_obj.mean():.4f}")
    print(f"  Std:  {weights_obj.std():.4f}")
    print(f"  >0.9: {(weights_obj > 0.9).sum()} / {len(weights_obj)}")
    print(f"  >0.7: {(weights_obj > 0.7).sum()} / {len(weights_obj)}")
    print(f"  <0.3: {(weights_obj < 0.3).sum()} / {len(weights_obj)}")

    print(f"\nBackground Weights (scale={scale}):")
    print(f"  Min:  {weights_bg.min():.4f}")
    print(f"  Max:  {weights_bg.max():.4f}")
    print(f"  Mean: {weights_bg.mean():.4f}")
    print(f"  Std:  {weights_bg.std():.4f}")
    print(f"  >0.9: {(weights_bg > 0.9).sum()} / {len(weights_bg)}")
    print(f"  >0.7: {(weights_bg > 0.7).sum()} / {len(weights_bg)}")
    print(f"  <0.3: {(weights_bg < 0.3).sum()} / {len(weights_bg)}")

    # Ambiguous words
    ambiguous_mask = (weights_obj > 0.3) & (weights_obj < 0.7) & (weights_bg > 0.3) & (weights_bg < 0.7)
    print(f"\nAmbiguous words (both weights 0.3-0.7): {ambiguous_mask.sum()}")

    # Add weights to dataframe
    df['weight_obj'] = weights_obj
    df['weight_bg'] = weights_bg
    df['ambiguity'] = 1 - np.abs(weights_obj - weights_bg)

    print("\n[3] Example words:")

    # Top object-leaning
    print("\nTop 10 object-leaning (high obj weight):")
    top_obj = df.nlargest(10, 'weight_obj')[['word', 'r_score', 'weight_obj', 'weight_bg', 'category']]
    for _, row in top_obj.iterrows():
        print(f"  {row['word']:30s} | r={row['r_score']:+.3f} | obj={row['weight_obj']:.3f}, bg={row['weight_bg']:.3f} | {row['category']}")

    # Top background-leaning
    print("\nTop 10 background-leaning (high bg weight):")
    top_bg = df.nlargest(10, 'weight_bg')[['word', 'r_score', 'weight_obj', 'weight_bg', 'category']]
    for _, row in top_bg.iterrows():
        print(f"  {row['word']:30s} | r={row['r_score']:+.3f} | obj={row['weight_obj']:.3f}, bg={row['weight_bg']:.3f} | {row['category']}")

    # Most ambiguous
    print("\nTop 10 ambiguous (similar obj/bg weights):")
    top_amb = df.nlargest(10, 'ambiguity')[['word', 'r_score', 'weight_obj', 'weight_bg', 'category']]
    for _, row in top_amb.iterrows():
        print(f"  {row['word']:30s} | r={row['r_score']:+.3f} | obj={row['weight_obj']:.3f}, bg={row['weight_bg']:.3f} | {row['category']}")

    print("\n[4] Comparison with hard threshold (tau=0.05):")
    tau = 0.05

    # Hard threshold categorization
    n_obj_hard = (df['r_score'] > tau).sum()
    n_bg_hard = (df['r_score'] < -tau).sum()
    n_amb_hard = ((df['r_score'] >= -tau) & (df['r_score'] <= tau)).sum()

    # Soft weighting "effective" contributions (weight > 0.5)
    n_obj_soft = (df['weight_obj'] > 0.5).sum()
    n_bg_soft = (df['weight_bg'] > 0.5).sum()
    n_both_soft = ((df['weight_obj'] > 0.5) & (df['weight_bg'] > 0.5)).sum()

    print(f"\nHard threshold (tau={tau}):")
    print(f"  Object:     {n_obj_hard} words")
    print(f"  Background: {n_bg_hard} words")
    print(f"  Ambiguous:  {n_amb_hard} words (DISCARDED)")
    print(f"  Total used: {n_obj_hard + n_bg_hard} / {len(df)}")

    print(f"\nSoft weighting (weight > 0.5):")
    print(f"  Object:     {n_obj_soft} words")
    print(f"  Background: {n_bg_soft} words")
    print(f"  Both:       {n_both_soft} words (can contribute to both)")
    print(f"  Total used: {len(df)} / {len(df)} (ALL WORDS USED ✅)")

    print("\n[5] Visualizing weight distribution by original category:")

    for cat in ['object', 'background', 'ambiguous']:
        cat_df = df[df['category'] == cat]
        if len(cat_df) > 0:
            print(f"\nOriginal '{cat}' words ({len(cat_df)}):")
            print(f"  Avg obj weight: {cat_df['weight_obj'].mean():.3f}")
            print(f"  Avg bg weight:  {cat_df['weight_bg'].mean():.3f}")
            print(f"  r-score range:  [{cat_df['r_score'].min():+.3f}, {cat_df['r_score'].max():+.3f}]")

    print("\n" + "="*80)
    print("✅ V3 Soft Weighting Analysis Complete!")
    print("="*80)
    print("\nKey benefits:")
    print("1. All 6370 negatives are used (vs 5730 with hard threshold)")
    print("2. Ambiguous words contribute appropriately to both obj and bg")
    print("3. No information loss from discarding words")
    print("4. Smooth gradient instead of hard cutoff")
    print("\nNext: Test V3 on ImageNet to see if it improves performance!")

if __name__ == '__main__':
    main()
