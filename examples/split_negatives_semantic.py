"""
Split negative labels into object-leaning and background-leaning sets
using CLIP text embeddings and template-based comparison.

This split is role-based, not a semantic ground-truth classification.
It supports object-centric base scoring and background-centric calibration
in spurious OOD detection.
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import clip
from tqdm import tqdm


def load_word_lists(noun_path: str, adj_path: str) -> List[str]:
    """
    Load and merge noun and adjective negative labels.
    Remove duplicates and return unique word list.
    """
    words = set()
    
    # Load nouns
    with open(noun_path, 'r') as f:
        for line in f:
            word = line.strip()
            if word:
                words.add(word)
    
    # Load adjectives
    with open(adj_path, 'r') as f:
        for line in f:
            word = line.strip()
            if word:
                words.add(word)
    
    return sorted(list(words))


def compute_word_embedding(word: str, model, device) -> torch.Tensor:
    """
    Compute CLIP embedding for a word using object-context phrase.
    
    Args:
        word: The word to encode
        model: CLIP model
        device: torch device
        
    Returns:
        L2-normalized embedding tensor
    """
    text = f"a photo of a {word}"
    tokens = clip.tokenize([text]).to(device)
    
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.cpu()


def compute_object_template_embedding(word: str, model, device) -> torch.Tensor:
    """
    Compute mean CLIP embedding from object-centric templates.
    
    Templates:
    - "a centered photo of a {}"
    - "a close-up photo of a {}"
    
    Args:
        word: The word to insert into templates
        model: CLIP model
        device: torch device
        
    Returns:
        L2-normalized mean embedding tensor
    """
    templates = [
        f"a centered photo of a {word}",
        f"a close-up photo of a {word}",
    ]
    
    tokens = clip.tokenize(templates).to(device)
    
    with torch.no_grad():
        embeddings = model.encode_text(tokens)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        mean_embedding = embeddings.mean(dim=0, keepdim=True)
        mean_embedding = mean_embedding / mean_embedding.norm(dim=-1, keepdim=True)
    
    return mean_embedding.cpu()


def compute_background_template_embedding(word: str, model, device) -> torch.Tensor:
    """
    Compute mean CLIP embedding from background-centric templates.
    
    Templates:
    - "a photo of a scene with {}"
    - "a background in a {} scene"
    - "a texture pattern of {}"
    
    Args:
        word: The word to insert into templates
        model: CLIP model
        device: torch device
        
    Returns:
        L2-normalized mean embedding tensor
    """
    templates = [
        f"a photo of a scene with {word}",
        f"a background in a {word} scene",
        f"a texture pattern of {word}",
    ]
    
    tokens = clip.tokenize(templates).to(device)
    
    with torch.no_grad():
        embeddings = model.encode_text(tokens)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        mean_embedding = embeddings.mean(dim=0, keepdim=True)
        mean_embedding = mean_embedding / mean_embedding.norm(dim=-1, keepdim=True)
    
    return mean_embedding.cpu()


def compute_objectness_score(word: str, model, device) -> float:
    """
    Compute objectness score for a word.
    
    r(w) = cos(e(w), E_obj(w)) - cos(e(w), E_bg(w))
    
    - r(w) > 0 : word aligns more with object usage
    - r(w) < 0 : word aligns more with background usage
    
    Args:
        word: The word to score
        model: CLIP model
        device: torch device
        
    Returns:
        Objectness score (float)
    """
    # Compute word embedding
    e_w = compute_word_embedding(word, model, device)
    
    # Compute object template embedding
    e_obj = compute_object_template_embedding(word, model, device)
    
    # Compute background template embedding
    e_bg = compute_background_template_embedding(word, model, device)
    
    # Compute cosine similarities
    cos_obj = (e_w * e_obj).sum().item()
    cos_bg = (e_w * e_bg).sum().item()
    
    # Objectness score
    r_score = cos_obj - cos_bg
    
    return r_score


def split_negatives(
    words: List[str],
    model,
    device,
    neg_object_ratio: float = 0.35
) -> Tuple[List[str], List[str], List[Tuple[str, float]]]:
    """
    Split negative vocabulary into object-leaning and background-leaning sets.
    
    Args:
        words: List of words to split
        model: CLIP model
        device: torch device
        neg_object_ratio: Fraction of words to assign as object-leaning (default 0.35)
        
    Returns:
        Tuple of (object_words, background_words, word_scores)
        where word_scores is list of (word, r_score) tuples
    """
    print(f"Computing objectness scores for {len(words)} words...")
    
    # Compute objectness scores for all words
    word_scores = []
    for word in tqdm(words, desc="Processing words"):
        r_score = compute_objectness_score(word, model, device)
        word_scores.append((word, r_score))
    
    # Sort by r_score in descending order
    word_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Determine threshold using percentile
    n_object = int(len(words) * neg_object_ratio)
    tau = word_scores[n_object - 1][1] if n_object > 0 else 0.0
    
    print(f"\nSplitting with threshold TAU = {tau:.6f}")
    print(f"Target object ratio: {neg_object_ratio:.2%}")
    
    # Split into object and background
    object_words = []
    background_words = []
    
    for word, r_score in word_scores:
        if r_score >= tau:
            object_words.append(word)
        else:
            background_words.append(word)
    
    return object_words, background_words, word_scores


def save_results(
    object_words: List[str],
    background_words: List[str],
    word_scores: List[Tuple[str, float]],
    output_dir: Path
):
    """
    Save split results to files.
    
    Files created:
    - neg_object.txt
    - neg_background.txt
    - neg_ambiguous.txt (empty for compatibility)
    - neg_word_scores.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save object words
    object_path = output_dir / "neg_object.txt"
    with open(object_path, 'w') as f:
        for word in object_words:
            f.write(f"{word}\n")
    print(f"Saved {len(object_words)} object-leaning words to {object_path}")
    
    # Save background words
    background_path = output_dir / "neg_background.txt"
    with open(background_path, 'w') as f:
        for word in background_words:
            f.write(f"{word}\n")
    print(f"Saved {len(background_words)} background-leaning words to {background_path}")
    
    # Save empty ambiguous file for compatibility
    ambiguous_path = output_dir / "neg_ambiguous.txt"
    with open(ambiguous_path, 'w') as f:
        pass  # Empty file
    print(f"Saved empty ambiguous file to {ambiguous_path}")
    
    # Save scores
    scores_path = output_dir / "neg_word_scores.csv"
    with open(scores_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'r_score'])
        for word, r_score in word_scores:
            writer.writerow([word, f"{r_score:.6f}"])
    print(f"Saved word scores to {scores_path}")


def print_summary(
    object_words: List[str],
    background_words: List[str],
    word_scores: List[Tuple[str, float]],
    tau: float
):
    """Print summary statistics."""
    total = len(object_words) + len(background_words)
    
    print("\n" + "=" * 80)
    print("SUMMARY: Negative Label Split")
    print("=" * 80)
    print(f"Total words:              {total}")
    print(f"Object-leaning words:     {len(object_words)} ({len(object_words)/total*100:.1f}%)")
    print(f"Background-leaning words: {len(background_words)} ({len(background_words)/total*100:.1f}%)")
    print(f"Threshold (TAU):          {tau:.6f}")
    print()
    
    # Show top 10 object-leaning words
    print("Top 10 object-leaning words:")
    for i, (word, score) in enumerate(word_scores[:10], 1):
        print(f"  {i:2d}. {word:40s} (r={score:.4f})")
    print()
    
    # Show top 10 background-leaning words (bottom of sorted list)
    print("Top 10 background-leaning words:")
    for i, (word, score) in enumerate(word_scores[-10:], 1):
        print(f"  {i:2d}. {word:40s} (r={score:.4f})")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Split negative labels into object and background sets using CLIP embeddings"
    )
    parser.add_argument(
        "--noun_path",
        type=str,
        default="data/neg_labels_noun.txt",
        help="Path to noun negative labels file"
    )
    parser.add_argument(
        "--adj_path",
        type=str,
        default="data/neg_labels_adj.txt",
        help="Path to adjective negative labels file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/split",
        help="Output directory for split files"
    )
    parser.add_argument(
        "--neg_object_ratio",
        type=float,
        default=0.35,
        help="Fraction of words to assign as object-leaning (default: 0.35)"
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B/16",
        help="CLIP model to use (default: ViT-B/16)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("Negative Label Splitting with CLIP Embeddings")
    print("=" * 80)
    print(f"Noun file:        {args.noun_path}")
    print(f"Adjective file:   {args.adj_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Object ratio:     {args.neg_object_ratio:.2%}")
    print(f"CLIP model:       {args.clip_model}")
    print(f"Device:           {args.device}")
    print(f"Random seed:      {args.seed}")
    print("=" * 80)
    print()
    
    # Load CLIP model
    print("Loading CLIP model...")
    model, preprocess = clip.load(args.clip_model, device=args.device)
    model.eval()
    print(f"✓ CLIP model loaded: {args.clip_model}")
    print()
    
    # Load word lists
    print("Loading negative labels...")
    words = load_word_lists(args.noun_path, args.adj_path)
    print(f"✓ Loaded {len(words)} unique words")
    print()
    
    # Split negatives
    object_words, background_words, word_scores = split_negatives(
        words=words,
        model=model,
        device=args.device,
        neg_object_ratio=args.neg_object_ratio
    )
    
    # Determine tau
    n_object = int(len(words) * args.neg_object_ratio)
    tau = word_scores[n_object - 1][1] if n_object > 0 else 0.0
    
    # Save results
    print("\nSaving results...")
    output_dir = Path(args.output_dir)
    save_results(object_words, background_words, word_scores, output_dir)
    
    # Print summary
    print_summary(object_words, background_words, word_scores, tau)
    
    print("\n✓ Splitting complete!")


if __name__ == "__main__":
    main()
