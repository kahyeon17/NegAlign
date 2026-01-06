#!/usr/bin/env python3
"""
NegAlign: Background-Bias-Aware Zero-Shot OOD Detection

Builds on NegRefine infrastructure with:
  - S_NegLabel* (modified base score with configurable enhancements)
  - P_align (CAM-based background alignment, NO SAM)

Final score: S_final = S_NegLabel* + lambda * P_align
"""

import clip
import torch
import pickle
import time
import os
import sys
import random
import numpy as np
from PIL import Image

# Add local utils directory to path
utils_dir = os.path.join(os.path.dirname(__file__), '..', 'utils')
core_dir = os.path.dirname(__file__)
sys.path.insert(0, utils_dir)
sys.path.insert(0, core_dir)

from class_names import CLASS_NAME, preset_noun_prompt_templates, preset_adj_prompt_templates
from create_negs import create_initial_negative_labels

# Import our modules
from split_negatives_by_clip import split_negatives, load_neg_labels
from cam_bg import ClipViTGradCAM, extract_bg_embedding


def _save_labels(labels, file_name):
    """Save labels to txt and pkl files."""
    with open(file_name + '.txt', 'w') as f:
        for w in labels:
            f.write(w + '\n')

    with open(file_name + '.pkl', 'wb') as fp:
        pickle.dump(labels, fp)


def _load_labels(file_name):
    """Load labels from pkl file."""
    if not os.path.exists(file_name + '.pkl'):
        raise FileNotFoundError(f"Label file not found: {file_name}.pkl")

    with open(file_name + '.pkl', 'rb') as fp:
        labels = pickle.load(fp)
    return labels


class CLIPNegAlign:
    """
    NegAlign: Minimal modification of NegRefine for background-bias-aware OOD detection.
    """

    def __init__(self,
                 train_dataset='imagenet',
                 arch='ViT-B/16',
                 seed=0,
                 device='cuda:0',
                 output_folder='./data/',
                 load_saved_labels=True,
                 # Negative split args
                 neg_split_dir=None,
                 neg_split_tau=0.05,
                 neg_use_ambiguous_in_obj=False,
                 neg_split_recompute=False,
                 # NegLabel* modification flags
                 use_neglabel_star=False,
                 topk_aggregation_k=10,
                 use_role_aware_negatives=False,
                 use_scale_stabilization=False,
                 # P_align args
                 use_p_align=False,
                 lambda_bg=1.0,
                 pos_topk=10,
                 neg_topk=5,
                 # CAM args
                 cam_fg_percentile=80,
                 cam_dilate_px=1,
                 cam_block=-1):

        self.device = device
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)

        # Load CLIP
        self.clip_model, self.clip_preprocess = clip.load(arch, device, jit=False)
        self.clip_model.eval()

        # Convert to float32 for CAM gradient computation
        if use_p_align:
            self.clip_model.float()

        # NegLabel* flags
        self.use_neglabel_star = use_neglabel_star
        self.topk_aggregation_k = topk_aggregation_k
        self.use_role_aware_negatives = use_role_aware_negatives
        self.use_scale_stabilization = use_scale_stabilization

        # P_align flags
        self.use_p_align = use_p_align
        self.lambda_bg = lambda_bg
        self.pos_topk = pos_topk
        self.neg_topk = neg_topk

        # CAM settings
        self.cam_fg_percentile = cam_fg_percentile
        self.cam_dilate_px = cam_dilate_px
        self.cam_block = cam_block

        # Initialize CAM generator if using P_align
        if self.use_p_align:
            self.cam_generator = ClipViTGradCAM(
                self.clip_model,
                target_layer_idx=cam_block,
                device=device
            )
            print(f"CAM initialized: block={cam_block}, fg_percentile={cam_fg_percentile}, dilate={cam_dilate_px}")
        else:
            self.cam_generator = None

        # Get class names and templates
        class_name = CLASS_NAME[train_dataset]
        self.pos_labels = class_name

        if train_dataset == 'imagenet_sketch':
            self.noun_prompt_templates = preset_noun_prompt_templates_for_sketch
        else:
            from class_names import preset_noun_prompt_templates
            self.noun_prompt_templates = preset_noun_prompt_templates

        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        self.output_folder = output_folder

        # ========================================================
        # NEGATIVE LABEL LOADING AND SPLITTING
        # ========================================================

        # Load initial negative labels (skip NegFilter)
        if load_saved_labels and os.path.exists(output_folder + 'neg_labels_noun.pkl'):
            print(f"\n--- Loading negative labels from: {output_folder}")
            neg_labels_noun = _load_labels(output_folder + 'neg_labels_noun')
            neg_labels_adj = _load_labels(output_folder + 'neg_labels_adj')
        else:
            print("\n--- Creating initial negative labels (CSP, no NegFilter)")
            start_time = time.time()
            neg_labels_noun, neg_labels_adj = create_initial_negative_labels(
                self.clip_model,
                train_dataset=train_dataset,
                neg_top_p=0.15,
                seed=seed,
                device=device
            )
            print(f"Time: {time.time() - start_time:.2f}s")
            print(f"Initial negatives: nouns={len(neg_labels_noun)}, adjs={len(neg_labels_adj)}")

            _save_labels(neg_labels_noun, output_folder + 'neg_labels_noun')
            _save_labels(neg_labels_adj, output_folder + 'neg_labels_adj')

        # Split negatives into obj/bg
        if neg_split_dir is not None:
            split_dir = neg_split_dir
            print(f"\n--- Loading split negatives from custom directory: {split_dir}")
        else:
            split_dir = os.path.join(output_folder, 'negatives_split')
            noun_file = output_folder + 'neg_labels_noun.txt'
            adj_file = output_folder + 'neg_labels_adj.txt'

            if neg_split_recompute or not os.path.exists(os.path.join(split_dir, 'neg_word_scores.csv')):
                print(f"\n--- Splitting negatives (tau={neg_split_tau})")
                split_negatives(
                    noun_file=noun_file,
                    adj_file=adj_file,
                    output_dir=split_dir,
                    tau=neg_split_tau,
                    device=device,
                    clip_arch=arch,
                    recompute=neg_split_recompute
                )
            else:
                print(f"\n--- Loading split negatives from: {split_dir}")

        # Load split negatives
        with open(os.path.join(split_dir, 'neg_object.txt'), 'r') as f:
            neg_obj_words = [line.strip() for line in f if line.strip()]

        with open(os.path.join(split_dir, 'neg_background.txt'), 'r') as f:
            neg_bg_words = [line.strip() for line in f if line.strip()]

        with open(os.path.join(split_dir, 'neg_ambiguous.txt'), 'r') as f:
            neg_amb_words = [line.strip() for line in f if line.strip()]

        # Decide which negatives to use for N_obj
        if neg_use_ambiguous_in_obj:
            N_obj = neg_obj_words + neg_amb_words
        else:
            N_obj = neg_obj_words

        N_bg = neg_bg_words

        print(f"\nNegative vocabulary:")
        print(f"  N_obj (base scoring): {len(N_obj)}")
        print(f"  N_bg (P_align):       {len(N_bg)}")
        print(f"  N_amb:                {len(neg_amb_words)} ({'included' if neg_use_ambiguous_in_obj else 'excluded'})")

        # ========================================================
        # EMBED TEXT LABELS
        # ========================================================

        print("\n--- Embedding text labels")

        # Positive features (all ID classes)
        self.pos_features = self._embed_text_labels(
            labels=self.pos_labels,
            prompt_templates=self.noun_prompt_templates
        )

        # Negative features: maintain separate sets for plain vs star
        all_negs = neg_labels_noun + neg_labels_adj

        # S_NegLabel_plain: always uses all negatives
        self.neg_features_plain = self._embed_text_labels(
            labels=all_negs,
            prompt_templates=self.noun_prompt_templates
        )
        print(f"  Plain negatives (all): {len(all_negs)} words")

        # S_NegLabel*: uses role-aware negatives if enabled
        if self.use_role_aware_negatives:
            self.neg_features_star = self._embed_text_labels(
                labels=N_obj,
                prompt_templates=self.noun_prompt_templates
            )
            print(f"  Star negatives (N_obj): {len(N_obj)} words")
        else:
            # If role-aware disabled, star uses same as plain
            self.neg_features_star = self.neg_features_plain
            print(f"  Star negatives (same as plain): {len(all_negs)} words")

        # Background negative features for P_align
        if self.use_p_align and len(N_bg) > 0:
            self.neg_features_bg = self._embed_text_labels(
                labels=N_bg,
                prompt_templates=self.noun_prompt_templates
            )
            print(f"  Background negatives: {len(N_bg)} words")
        else:
            self.neg_features_bg = None

        print(f"\nFeature shapes:")
        print(f"  pos_features:       {self.pos_features.shape}")
        print(f"  neg_features_plain: {self.neg_features_plain.shape}")
        print(f"  neg_features_star:  {self.neg_features_star.shape}")
        if self.neg_features_bg is not None:
            print(f"  neg_features_bg:    {self.neg_features_bg.shape}")

        # NegLabel grouping parameters
        self.ngroup = 100
        self.group_fuse_num = None

    def _embed_text_labels(self, labels, prompt_templates, batch_size=1000):
        """Embed text labels using CSP templates (average over prompts)."""
        texts = []
        for w in labels:
            for template in prompt_templates:
                texts.append(template.format(w))

        tokenized = torch.cat([clip.tokenize(f"{c}") for c in texts]).to(self.device)

        with torch.no_grad():
            features = []
            for i in range(0, len(tokenized), batch_size):
                x = self.clip_model.encode_text(tokenized[i: i + batch_size])
                features.append(x)
            features = torch.cat(features, dim=0)

            # Average over prompts
            features = features.view(-1, len(prompt_templates), features.shape[-1]).mean(dim=1)
            features /= features.norm(dim=-1, keepdim=True)
            features = features.to(torch.float32)

        return features

    def _grouping(self, pos, neg, num, ngroup=10, random_permute=False):
        """NegRefine grouping logic (softmax-based)."""
        group = ngroup
        drop = neg.shape[1] % ngroup
        if drop > 0:
            neg = neg[:, :-drop]

        if random_permute:
            torch.manual_seed(self.seed)
            idx = torch.randperm(neg.shape[1], device=self.device)
            neg = neg.T
            negs = neg[idx].T.reshape(pos.shape[0], group, -1).contiguous()
        else:
            negs = neg.reshape(pos.shape[0], group, -1).contiguous()

        scores = []
        for i in range(group):
            full_sim = torch.cat([pos, negs[:, i, :]], dim=-1)
            full_sim = full_sim.softmax(dim=-1)
            pos_score = full_sim[:, :pos.shape[1]].sum(dim=-1)
            scores.append(pos_score.unsqueeze(-1))

        scores = torch.cat(scores, dim=-1)
        if num is not None:
            scores = scores[:, 0:num - 1]
        score = scores.mean(dim=-1)

        return score

    def _neg_label_score_plain(self, pos_sim, neg_sim):
        """
        S_NegLabel_plain: Original NegRefine base score.
        Softmax-based grouping (no modifications).
        """
        if self.ngroup > 1:
            score = self._grouping(pos_sim, neg_sim, num=self.group_fuse_num, ngroup=self.ngroup, random_permute=True)
        else:
            full_sim = torch.cat([pos_sim, neg_sim], dim=-1)
            full_sim = full_sim.softmax(dim=-1)
            pos_score = full_sim[:, :pos_sim.shape[1]].sum(dim=1)
            score = pos_score

        return score

    def _neg_label_score_star(self, pos_sim, neg_sim):
        """
        S_NegLabel*: Modified base score with optional enhancements.

        Modifications (configurable):
          (M1) topk_aggregation: Replace max with TopKMean
          (M2) role_aware_negatives: Use N_obj only (handled in __init__)
          (M3) scale_stabilization: Z-score normalization (optional)
        """
        # Start with plain score
        score = self._neg_label_score_plain(pos_sim, neg_sim)

        # (M1) TopK aggregation (if enabled)
        # Note: Current implementation uses grouping/mean, which is already an aggregation.
        # For explicit TopK, we would need to modify the grouping logic.
        # For now, we keep the grouping as-is (it's already a form of mean aggregation).

        # (M3) Scale stabilization (optional)
        if self.use_scale_stabilization:
            # Z-score normalization within batch
            score = (score - score.mean()) / (score.std() + 1e-8)

        return score

    def _compute_p_align(self, image_tensor, orig_image, predicted_class_idx):
        """
        Compute P_align using CAM-based background embedding.

        Returns:
            p_align: float
            cam_valid: bool
        """
        if not self.use_p_align or self.cam_generator is None:
            return 0.0, False

        # Get text feature for predicted class
        text_feature_c_hat = self.pos_features[predicted_class_idx]

        # Extract background embedding via CAM
        bg_embedding, cam_valid = extract_bg_embedding(
            self.clip_model,
            image_tensor,
            text_feature_c_hat,
            self.cam_generator,
            fg_percentile=self.cam_fg_percentile,
            dilate_px=self.cam_dilate_px
        )

        if not cam_valid or self.neg_features_bg is None:
            return 0.0, False

        # S_bg_pos: TopKMean similarity with positive prompts for c_hat
        # For simplicity, we use the already-embedded positive feature for c_hat
        # (which is averaged over CSP templates)
        s_bg_pos = float((bg_embedding @ text_feature_c_hat).item())

        # S_bg_neg: TopKMean similarity with background negatives
        sim_neg_all = bg_embedding @ self.neg_features_bg.T  # (M_bg,)
        topk_neg = torch.topk(sim_neg_all, k=min(self.neg_topk, len(sim_neg_all)), largest=True)
        s_bg_neg = float(topk_neg.values.mean().item())

        # P_align = S_bg_pos - S_bg_neg
        # If background is aligned with predicted class (high S_bg_pos), it's likely ID → positive P_align
        # If background is aligned with N_bg words (high S_bg_neg), it's likely OOD → negative P_align
        # We SUBTRACT P_align from S_NegLabel*, so negative P_align lowers OOD score
        p_align = s_bg_pos - s_bg_neg

        return p_align, True

    def detection_score(self, img, orig_image=None, return_details=False):
        """
        Compute NegAlign detection score.

        Args:
            img: (1, 3, H, W) preprocessed CLIP image tensor
            orig_image: PIL Image (not used, for API compatibility)
            return_details: if True, return dict with components

        Returns:
            If return_details=True:
                dict with 's_neglabel_plain', 's_neglabel_star', 'p_align', 'cam_valid', 's_final'
            Else:
                s_final: float
        """
        with torch.no_grad():
            # Encode image
            image_features = self.clip_model.encode_image(img)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.to(torch.float32)

            # Compute similarities (scale by 100 as in NegRefine)
            pos_sim = (100.0 * image_features @ self.pos_features.T)

            # S_NegLabel_plain: uses ALL negatives
            neg_sim_plain = (100.0 * image_features @ self.neg_features_plain.T)
            s_neglabel_plain = float(self._neg_label_score_plain(pos_sim, neg_sim_plain).item())

            # S_NegLabel*: ALWAYS computed using neg_features_star (role-aware or full)
            neg_sim_star = (100.0 * image_features @ self.neg_features_star.T)
            s_neglabel_star = float(self._neg_label_score_star(pos_sim, neg_sim_star).item())

            # Get predicted class
            predicted_class_idx = pos_sim.argmax().item()

        # P_align (if enabled)
        if self.use_p_align:
            p_align, cam_valid = self._compute_p_align(img, orig_image, predicted_class_idx)
        else:
            p_align = 0.0
            cam_valid = False

        # Final score: use plain or star based on flag
        if self.use_neglabel_star:
            base_score = s_neglabel_star
        else:
            base_score = s_neglabel_plain
        
        # Add P_align (original design)
        # Positive P_align (bg aligned with class) → ID-like → higher score
        # Negative P_align (bg aligned with N_bg) → OOD-like → lower score
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


# ========================================================
# SANITY CHECK FUNCTION
# ========================================================

def sanity_check(model, image_path=None):
    """
    Run sanity check on a single image.

    Prints: c_hat, S_NegLabel_plain, S_NegLabel*, S_bg_pos, S_bg_neg, P_align, S_final
    """
    from PIL import Image

    # Load or create test image
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert('RGB')
    else:
        # Create solid color test image
        img = Image.new('RGB', (224, 224), color='red')

    # Preprocess
    img_tensor = model.clip_preprocess(img).unsqueeze(0).to(model.device)

    # Get detection score with details
    details = model.detection_score(img_tensor, orig_image=img, return_details=True)

    # Print results
    print("\n" + "="*60)
    print("SANITY CHECK RESULTS")
    print("="*60)
    print(f"Predicted class idx: {details['predicted_class_idx']}")
    print(f"Predicted class:     {model.pos_labels[details['predicted_class_idx']]}")
    print(f"\nS_NegLabel_plain:    {details['s_neglabel_plain']:.4f}")
    print(f"S_NegLabel*:         {details['s_neglabel_star']:.4f}")
    print(f"\nP_align:             {details['p_align']:.4f}")
    print(f"CAM valid:           {details['cam_valid']}")
    print(f"\nS_final:             {details['s_final']:.4f}")
    print(f"  = S_NegLabel* + {model.lambda_bg} * P_align")
    print(f"  = {details['s_neglabel_star']:.4f} + {model.lambda_bg} * {details['p_align']:.4f}")
    print("="*60)

    return details


if __name__ == '__main__':
    print("Initializing NegAlign model...")

    model = CLIPNegAlign(
        train_dataset='imagenet',
        arch='ViT-B/16',
        seed=0,
        device='cuda:1',
        output_folder='./data/',
        load_saved_labels=True,
        # NegLabel* modifications
        use_neglabel_star=True,
        topk_aggregation_k=10,
        use_role_aware_negatives=True,
        use_scale_stabilization=False,
        # P_align
        use_p_align=True,
        lambda_bg=1.0,
        pos_topk=10,
        neg_topk=5,
        # CAM
        cam_fg_percentile=80,
        cam_dilate_px=1,
        cam_block=-1
    )

    print("\nRunning sanity check...")
    sanity_check(model)
