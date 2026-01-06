#!/usr/bin/env python3
"""
Split negative labels into object-leaning vs background-leaning using CLIP text embeddings.
No POS taggers, no hand-written rules - purely embedding-based classification.
"""

import os
import sys
import csv
import torch
import clip
import argparse
from tqdm import tqdm


# Object templates: emphasize concrete objects
OBJECT_TEMPLATES = [
    "a photo of a {}",
    "a close-up photo of a {}",
]

# Background templates: emphasize scene/texture/context
BACKGROUND_TEMPLATES = [
    "a scene of {}",
    "a background of {}",
    "a texture of {}",
]


def load_neg_labels(noun_file, adj_file):
    """Load negative labels from txt files."""
    words = []

    if os.path.exists(noun_file):
        with open(noun_file, 'r') as f:
            words.extend([line.strip() for line in f if line.strip()])

    if os.path.exists(adj_file):
        with open(adj_file, 'r') as f:
            words.extend([line.strip() for line in f if line.strip()])

    return words


def embed_word_with_templates(word, templates, clip_model, device):
    """Encode word with multiple templates and return average embedding."""
    texts = [template.format(word) for template in templates]
    tokens = torch.cat([clip.tokenize(text) for text in texts]).to(device)

    with torch.no_grad():
        features = clip_model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.to(torch.float32)
        avg_feature = features.mean(dim=0)
        avg_feature = avg_feature / avg_feature.norm()

    return avg_feature


def compute_r_score(word, clip_model, device):
    """
    Compute r(w) = cos(e(w), E_obj(w)) - cos(e(w), E_bg(w))

    Returns:
        r_score: float
    """
    # Raw word embedding (or "a {word}")
    raw_text = f"a {word}"
    raw_token = clip.tokenize([raw_text]).to(device)
    with torch.no_grad():
        e_w = clip_model.encode_text(raw_token)[0]
        e_w = e_w / e_w.norm()
        e_w = e_w.to(torch.float32)

    # Object-template embedding
    e_obj = embed_word_with_templates(word, OBJECT_TEMPLATES, clip_model, device)

    # Background-template embedding
    e_bg = embed_word_with_templates(word, BACKGROUND_TEMPLATES, clip_model, device)

    # Cosine similarities
    cos_obj = float((e_w @ e_obj).item())
    cos_bg = float((e_w @ e_bg).item())

    r_score = cos_obj - cos_bg

    return r_score


def classify_word(r_score, tau_obj=0.01, tau_bg=0.05):
    """
    Classify word into:
        - 'object': r_score > +tau_obj
        - 'background': r_score < -tau_bg
        - 'ambiguous': otherwise

    Uses asymmetric thresholds to compensate for background bias in negative labels.
    """
    if r_score > tau_obj:
        return 'object'
    elif r_score < -tau_bg:
        return 'background'
    else:
        return 'ambiguous'


def split_negatives(noun_file, adj_file, output_dir, tau=0.05, tau_obj=None, tau_bg=None,
                   percentile=None, device='cuda:0', clip_arch='ViT-B/16', recompute=False):
    """
    Main function to split negative labels.

    Args:
        noun_file: path to initial_neg_labels_noun.txt
        adj_file: path to initial_neg_labels_adj.txt
        output_dir: directory to save outputs
        tau: symmetric threshold (deprecated, use tau_obj and tau_bg)
        tau_obj: threshold for object classification (default: tau)
        tau_bg: threshold for background classification (default: tau)
        percentile: use percentile-based split (e.g., 30 for top/bottom 30%)
        device: CUDA device
        clip_arch: CLIP model architecture
        recompute: if True, recompute even if files exist
    """
    # Use asymmetric thresholds if specified, otherwise use symmetric tau
    if tau_obj is None:
        tau_obj = tau
    if tau_bg is None:
        tau_bg = tau
    os.makedirs(output_dir, exist_ok=True)

    csv_file = os.path.join(output_dir, 'neg_word_scores.csv')
    obj_file = os.path.join(output_dir, 'neg_object.txt')
    bg_file = os.path.join(output_dir, 'neg_background.txt')
    amb_file = os.path.join(output_dir, 'neg_ambiguous.txt')

    # Check if already exists
    if not recompute and all(os.path.exists(f) for f in [csv_file, obj_file, bg_file, amb_file]):
        print(f"Split files already exist in {output_dir}. Use --neg_split_recompute to recompute.")
        return

    print(f"Loading negative labels from:\n  {noun_file}\n  {adj_file}")
    words = load_neg_labels(noun_file, adj_file)
    print(f"Total words to classify: {len(words)}")

    # Load CLIP
    print(f"Loading CLIP model: {clip_arch}")
    clip_model, _ = clip.load(clip_arch, device=device, jit=False)
    clip_model.eval()

    # Compute r-scores
    if percentile:
        print(f"Computing r-scores (percentile={percentile}%)...")
    else:
        print(f"Computing r-scores (tau_obj={tau_obj}, tau_bg={tau_bg})...")

    results = []

    for word in tqdm(words, desc="Classifying words"):
        r_score = compute_r_score(word, clip_model, device)
        results.append({
            'word': word,
            'r_score': r_score,
            'category': 'temp'  # Will be updated if using percentile
        })

    # Percentile-based classification (overrides tau-based)
    if percentile:
        # Sort by r_score
        sorted_results = sorted(results, key=lambda x: x['r_score'], reverse=True)
        n_top = int(len(results) * percentile / 100)
        n_bottom = int(len(results) * percentile / 100)

        # Top percentile -> object
        for i in range(n_top):
            sorted_results[i]['category'] = 'object'

        # Bottom percentile -> background
        for i in range(len(sorted_results) - n_bottom, len(sorted_results)):
            sorted_results[i]['category'] = 'background'

        # Middle -> ambiguous
        for i in range(n_top, len(sorted_results) - n_bottom):
            sorted_results[i]['category'] = 'ambiguous'

        results = sorted_results
    else:
        # Tau-based classification
        for r in results:
            r['category'] = classify_word(r['r_score'], tau_obj=tau_obj, tau_bg=tau_bg)

    # Write CSV
    print(f"Writing results to {csv_file}")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['word', 'r_score', 'category'])
        writer.writeheader()
        writer.writerows(results)

    # Split into separate files
    obj_words = [r['word'] for r in results if r['category'] == 'object']
    bg_words = [r['word'] for r in results if r['category'] == 'background']
    amb_words = [r['word'] for r in results if r['category'] == 'ambiguous']

    with open(obj_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(obj_words))

    with open(bg_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(bg_words))

    with open(amb_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(amb_words))

    print(f"\n{'='*60}")
    print(f"Split Summary:")
    print(f"  Object-leaning:     {len(obj_words):5d} ({len(obj_words)/len(words)*100:.1f}%)")
    print(f"  Background-leaning: {len(bg_words):5d} ({len(bg_words)/len(words)*100:.1f}%)")
    print(f"  Ambiguous:          {len(amb_words):5d} ({len(amb_words)/len(words)*100:.1f}%)")
    print(f"  Total:              {len(words):5d}")
    print(f"{'='*60}")
    print(f"\nFiles saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split negative labels by CLIP embeddings')
    parser.add_argument('--noun_file', type=str, required=True,
                       help='Path to initial_neg_labels_noun.txt')
    parser.add_argument('--adj_file', type=str, required=True,
                       help='Path to initial_neg_labels_adj.txt')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save split results')
    parser.add_argument('--tau', type=float, default=0.05,
                       help='Symmetric threshold (default: 0.05)')
    parser.add_argument('--tau_obj', type=float, default=None,
                       help='Object threshold (default: same as --tau)')
    parser.add_argument('--tau_bg', type=float, default=None,
                       help='Background threshold (default: same as --tau)')
    parser.add_argument('--percentile', type=float, default=None,
                       help='Percentile-based split (e.g., 30 for top/bottom 30%%)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='CUDA device (default: cuda:0)')
    parser.add_argument('--clip_arch', type=str, default='ViT-B/16',
                       help='CLIP architecture (default: ViT-B/16)')
    parser.add_argument('--recompute', action='store_true',
                       help='Recompute even if files exist')

    args = parser.parse_args()

    split_negatives(
        noun_file=args.noun_file,
        adj_file=args.adj_file,
        output_dir=args.output_dir,
        tau=args.tau,
        tau_obj=args.tau_obj,
        tau_bg=args.tau_bg,
        percentile=args.percentile,
        device=args.device,
        clip_arch=args.clip_arch,
        recompute=args.recompute
    )
