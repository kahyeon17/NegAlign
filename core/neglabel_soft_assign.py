#!/usr/bin/env python3
"""
NegLabel with Soft Assignment of Ambiguous Negatives (Option B)

Extends CLIPNegAlign to support soft assignment of ambiguous negative labels
to both base scoring (S_NegLabel*) and background alignment (P_align).

Key Features:
- Backward compatible: --use_amb_soft=False reproduces existing behavior
- Soft weighting: ambiguous words weighted by p_obj(w) and p_bg(w) based on r_score
- Minimal code changes: extends existing CLIPNegAlign class
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import csv
import clip

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from clip_negalign import CLIPNegAlign
from cam_bg import extract_bg_embedding


class CLIPNegAlignSoftAssign(CLIPNegAlign):
    """
    Extended CLIPNegAlign with soft assignment of ambiguous negatives.
    
    Ambiguous negatives are distributed to both:
    - Base scoring (S_NegLabel*) with weight p_obj(w)
    - Background alignment (P_align) with weight p_bg(w)
    
    where weights are derived from r_score using sigmoid function.
    """
    
    def __init__(
        self,
        train_dataset='imagenet',
        arch='ViT-B/16',
        seed=0,
        device='cuda:0',
        output_folder='./data/',
        load_saved_labels=True,
        use_neglabel_star=False,
        use_role_aware_negatives=True,
        use_p_align=True,
        lambda_bg=1.0,
        pos_topk=10,
        neg_topk=5,
        cam_fg_percentile=80,
        cam_dilate_px=1,
        cam_block=-1,
        # New parameters for soft assignment
        use_amb_soft=False,
        amb_soft_beta=10.0
    ):
        """
        Initialize CLIPNegAlign with soft assignment option.
        
        Args:
            use_amb_soft (bool): Enable soft assignment of ambiguous negatives
            amb_soft_beta (float): Beta parameter for sigmoid weighting (default=10.0)
        """
        # Initialize parent class
        super().__init__(
            train_dataset=train_dataset,
            arch=arch,
            seed=seed,
            device=device,
            output_folder=output_folder,
            load_saved_labels=load_saved_labels,
            use_neglabel_star=use_neglabel_star,
            use_role_aware_negatives=use_role_aware_negatives,
            use_p_align=use_p_align,
            lambda_bg=lambda_bg,
            pos_topk=pos_topk,
            neg_topk=neg_topk,
            cam_fg_percentile=cam_fg_percentile,
            cam_dilate_px=cam_dilate_px,
            cam_block=cam_block
        )
        
        self.use_amb_soft = use_amb_soft
        self.amb_soft_beta = amb_soft_beta
        
        # Load ambiguous negatives with soft weights
        if self.use_amb_soft:
            self._load_ambiguous_soft_weights()
            self._embed_ambiguous_negatives()
            print(f"\n[Soft Assignment] Enabled with beta={amb_soft_beta}")
            print(f"  Ambiguous words: {len(self.amb_words)}")
            print(f"  Mean p_bg: {np.mean(list(self.amb_p_bg.values())):.4f}")
            print(f"  Mean p_obj: {np.mean(list(self.amb_p_obj.values())):.4f}")
    
    def _load_ambiguous_soft_weights(self):
        """Load ambiguous negatives and compute soft assignment weights."""
        neg_split_dir = Path(self.output_folder) / 'negatives_split'
        
        # Load ambiguous words
        amb_file = neg_split_dir / 'neg_ambiguous.txt'
        if not amb_file.exists():
            print(f"Warning: {amb_file} not found, soft assignment disabled")
            self.amb_words = []
            self.amb_p_bg = {}
            self.amb_p_obj = {}
            return
        
        with open(amb_file, 'r') as f:
            self.amb_words = [line.strip() for line in f if line.strip()]
        
        # Load r_scores from neg_word_scores.csv
        scores_file = neg_split_dir / 'neg_word_scores.csv'
        if not scores_file.exists():
            print(f"Warning: {scores_file} not found, using uniform weights")
            # Fallback: uniform weights (0.5 each)
            self.amb_p_bg = {w: 0.5 for w in self.amb_words}
            self.amb_p_obj = {w: 0.5 for w in self.amb_words}
            return
        
        # Parse CSV and extract r_scores for ambiguous words
        r_scores = {}
        with open(scores_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row['word']
                if word in self.amb_words:
                    r_scores[word] = float(row['r_score'])
        
        # Compute soft weights using sigmoid
        # p_bg(w) = sigmoid(-beta * r_score[w])
        # p_obj(w) = 1 - p_bg(w)
        self.amb_p_bg = {}
        self.amb_p_obj = {}
        
        for word in self.amb_words:
            if word in r_scores:
                r = r_scores[word]
                # sigmoid(-beta * r)
                p_bg = 1.0 / (1.0 + np.exp(self.amb_soft_beta * r))
                p_obj = 1.0 - p_bg
            else:
                # Fallback: uniform
                p_bg = 0.5
                p_obj = 0.5
            
            self.amb_p_bg[word] = p_bg
            self.amb_p_obj[word] = p_obj
    
    def _embed_ambiguous_negatives(self):
        """Embed ambiguous negative labels."""
        if not self.amb_words:
            self.amb_features = None
            self.amb_weights_obj = None
            self.amb_weights_bg = None
            return
        
        # Create text prompts
        amb_texts = [f"a photo of a {word}" for word in self.amb_words]
        amb_tokens = clip.tokenize(amb_texts).to(self.device)
        
        # Encode
        with torch.no_grad():
            amb_features = self.clip_model.encode_text(amb_tokens).float()
            amb_features /= amb_features.norm(dim=-1, keepdim=True)
        
        self.amb_features = amb_features  # (N_amb, D)
        
        # Store weights as tensors
        self.amb_weights_obj = torch.tensor(
            [self.amb_p_obj[w] for w in self.amb_words],
            dtype=torch.float32,
            device=self.device
        )  # (N_amb,)
        
        self.amb_weights_bg = torch.tensor(
            [self.amb_p_bg[w] for w in self.amb_words],
            dtype=torch.float32,
            device=self.device
        )  # (N_amb,)
        
        print(f"  Embedded {len(self.amb_words)} ambiguous negatives")
    
    def detection_score(self, img, orig_image=None, return_details=False):
        """
        Compute detection score with soft assignment of ambiguous negatives.
        
        Modified behavior when use_amb_soft=True:
        - S_NegLabel* includes ambiguous negatives weighted by p_obj(w)
        - P_align includes ambiguous negatives weighted by p_bg(w)
        """
        with torch.no_grad():
            # Extract CLIP features and ensure float32 dtype
            image_features = self.clip_model.encode_image(img).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Positive similarity
            pos_sim = (100.0 * image_features @ self.pos_features.T)
            
            # S_NegLabel_plain: uses ALL negatives (unchanged)
            neg_sim_plain = (100.0 * image_features @ self.neg_features_plain.T)
            s_neglabel_plain = float(self._neg_label_score_plain(pos_sim, neg_sim_plain).item())
            
            # S_NegLabel_star: uses role-aware negatives
            # MODIFIED: Add ambiguous negatives with soft weights
            if self.use_amb_soft and self.amb_features is not None:
                # Compute similarity with role-aware negatives
                neg_sim_star = (100.0 * image_features @ self.neg_features_star.T)
                
                # Compute similarity with ambiguous negatives
                amb_sim = (100.0 * image_features @ self.amb_features.T)  # (1, N_amb)
                
                # Apply soft weights: p_obj(w) * cos(I, e(w))
                amb_sim_weighted = amb_sim * self.amb_weights_obj.unsqueeze(0)  # (1, N_amb)
                
                # Combine: [neg_star, amb_weighted]
                combined_sim = torch.cat([neg_sim_star, amb_sim_weighted], dim=1)
                
                # Compute score using combined similarities
                s_neglabel_star = float(self._neg_label_score_star(pos_sim, combined_sim).item())
            else:
                # Original behavior
                neg_sim_star = (100.0 * image_features @ self.neg_features_star.T)
                s_neglabel_star = float(self._neg_label_score_star(pos_sim, neg_sim_star).item())
            
            # Get predicted class
            predicted_class_idx = pos_sim.argmax().item()
        
        # P_align (if enabled)
        # MODIFIED: Add ambiguous negatives with soft weights to background scoring
        if self.use_p_align:
            p_align, cam_valid = self._compute_p_align_soft(img, orig_image, predicted_class_idx)
        else:
            p_align = 0.0
            cam_valid = False
        
        # Final score: use plain or star based on flag
        if self.use_neglabel_star:
            base_score = s_neglabel_star
        else:
            base_score = s_neglabel_plain
        
        s_final = base_score + self.lambda_bg * p_align
        
        if return_details:
            return {
                's_neglabel_plain': s_neglabel_plain,
                's_neglabel_star': s_neglabel_star,
                'p_align': p_align,
                'cam_valid': cam_valid,
                's_final': s_final,
                'predicted_class_idx': predicted_class_idx
            }
        else:
            return s_final
    
    def _compute_p_align_soft(self, img, orig_image, predicted_class_idx):
        """
        Compute P_align with soft assignment of ambiguous negatives.
        
        MODIFIED: Background negative score includes ambiguous words weighted by p_bg(w).
        """
        if not self.use_p_align or self.cam_generator is None:
            return 0.0, False
        
        # Get text feature for predicted class
        text_feature_c_hat = self.pos_features[predicted_class_idx]
        
        # Extract background embedding via CAM (same as parent class)
        bg_embedding, cam_valid = extract_bg_embedding(
            self.clip_model,
            img,
            text_feature_c_hat,
            self.cam_generator,
            fg_percentile=self.cam_fg_percentile,
            dilate_px=self.cam_dilate_px
        )
        
        if not cam_valid or self.neg_features_bg is None:
            return 0.0, False
        
        # S_bg_pos: similarity with positive class
        s_bg_pos = float((bg_embedding @ text_feature_c_hat).item())
        
        # S_bg_neg: similarity with background negatives
        # MODIFIED: Add ambiguous negatives with soft weights
        if self.use_amb_soft and self.amb_features is not None:
            # Similarity with background negatives
            sim_bg_neg = bg_embedding @ self.neg_features_bg.T  # (N_bg,)
            
            # Similarity with ambiguous negatives
            sim_amb = bg_embedding @ self.amb_features.T  # (N_amb,)
            
            # Apply soft weights: p_bg(w) * cos(I_bg, e(w))
            sim_amb_weighted = sim_amb * self.amb_weights_bg  # (N_amb,)
            
            # Combine: [bg_neg, amb_weighted]
            sim_combined = torch.cat([sim_bg_neg, sim_amb_weighted])  # (N_bg + N_amb,)
            
            # Top-k mean over combined similarities
            topk_neg = torch.topk(sim_combined, k=min(self.neg_topk, len(sim_combined)), largest=True)
            s_bg_neg = float(topk_neg.values.mean().item())
        else:
            # Original behavior (same as parent class)
            sim_neg_all = bg_embedding @ self.neg_features_bg.T
            topk_neg = torch.topk(sim_neg_all, k=min(self.neg_topk, len(sim_neg_all)), largest=True)
            s_bg_neg = float(topk_neg.values.mean().item())
        
        # P_align = S_bg_pos - S_bg_neg
        p_align = s_bg_pos - s_bg_neg
        
        return p_align, True


def main():
    """Example usage of CLIPNegAlignSoftAssign."""
    import argparse
    
    parser = argparse.ArgumentParser(description='NegLabel with Soft Assignment of Ambiguous Negatives')
    parser.add_argument('--dataset', type=str, default='imagenet', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--use_amb_soft', action='store_true', help='Enable soft assignment')
    parser.add_argument('--amb_soft_beta', type=float, default=10.0, help='Beta for sigmoid weighting')
    
    args = parser.parse_args()
    
    print("="*80)
    print("NegLabel with Soft Assignment - Example Usage")
    print("="*80)
    
    # Initialize model
    model = CLIPNegAlignSoftAssign(
        train_dataset=args.dataset,
        arch='ViT-B/16',
        seed=0,
        device=args.device,
        output_folder='./data/',
        load_saved_labels=True,
        use_neglabel_star=False,
        use_role_aware_negatives=True,
        use_p_align=True,
        lambda_bg=1.0,
        use_amb_soft=args.use_amb_soft,
        amb_soft_beta=args.amb_soft_beta
    )
    
    print("\n" + "="*80)
    print("Model Configuration:")
    print("="*80)
    print(f"  use_amb_soft: {model.use_amb_soft}")
    if model.use_amb_soft:
        print(f"  amb_soft_beta: {model.amb_soft_beta}")
        print(f"  Ambiguous words: {len(model.amb_words)}")
        if model.amb_words:
            print(f"  Mean p_bg: {np.mean(list(model.amb_p_bg.values())):.4f}")
            print(f"  Mean p_obj: {np.mean(list(model.amb_p_obj.values())):.4f}")
    
    print("\n" + "="*80)
    print("Ready for OOD detection experiments")
    print("="*80)


if __name__ == "__main__":
    main()
