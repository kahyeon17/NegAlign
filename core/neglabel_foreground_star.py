#!/usr/bin/env python3
"""
Foreground-based S_NegLabel_star with P_align

Key idea:
- S_NegLabel_star: computed on FOREGROUND features only (CAM-based)
- P_align: computed on BACKGROUND features (CAM-based)

This separates object detection (foreground) from background bias detection.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from clip_negalign import CLIPNegAlign
from cam_bg import extract_bg_embedding


class CLIPNegAlignForegroundStar(CLIPNegAlign):
    """
    Extended CLIPNegAlign with foreground-based S_NegLabel_star.
    
    Architecture:
    - S_NegLabel_star: foreground features @ N_obj (object detection)
    - P_align: background features @ N_bg (background bias detection)
    
    This provides clearer separation of roles compared to using full image for both.
    """
    
    def __init__(
        self,
        train_dataset='imagenet',
        arch='ViT-B/16',
        seed=0,
        device='cuda:0',
        output_folder='./data/',
        load_saved_labels=True,
        use_role_aware_negatives=True,
        use_p_align=True,
        lambda_bg=1.0,
        pos_topk=10,
        neg_topk=5,
        cam_fg_percentile=80,
        cam_dilate_px=1,
        cam_block=-1,
        **kwargs
    ):
        """
        Initialize foreground-based NegAlign.
        
        Note: use_neglabel_star is forced to True (uses N_obj for foreground scoring).
        """
        # Force use_neglabel_star=True since we're using foreground
        super().__init__(
            train_dataset=train_dataset,
            arch=arch,
            seed=seed,
            device=device,
            output_folder=output_folder,
            load_saved_labels=load_saved_labels,
            use_neglabel_star=True,  # Always use star (N_obj)
            use_role_aware_negatives=use_role_aware_negatives,
            use_p_align=use_p_align,
            lambda_bg=lambda_bg,
            pos_topk=pos_topk,
            neg_topk=neg_topk,
            cam_fg_percentile=cam_fg_percentile,
            cam_dilate_px=cam_dilate_px,
            cam_block=cam_block,
            **kwargs
        )
        
        print(f"[Foreground-based Star] Enabled")
        print(f"  Foreground: object detection with N_obj ({len(self.neg_features_star)} words)")
        print(f"  Background: bias detection with N_bg ({len(self.neg_features_bg) if self.neg_features_bg is not None else 0} words)")
    
    def detection_score(self, img, orig_image=None, return_details=False):
        """
        Compute detection score with foreground-based S_NegLabel_star.
        
        Modified behavior:
        - S_NegLabel_star uses FOREGROUND features only
        - P_align uses BACKGROUND features (unchanged)
        """
        with torch.no_grad():
            # Get predicted class from full image (for CAM targeting)
            image_features = self.clip_model.encode_image(img)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.to(torch.float32)
            pos_sim = (100.0 * image_features @ self.pos_features.T)
            predicted_class_idx = pos_sim.argmax().item()
        
        # Extract foreground features via CAM
        if self.cam_generator is None:
            # Fallback: use full image if CAM not available
            s_neglabel_star = float(self._neg_label_score_star(pos_sim, 
                                    100.0 * image_features @ self.neg_features_star.T).item())
            p_align = 0.0
            cam_valid = False
        else:
            text_feature_c_hat = self.pos_features[predicted_class_idx]
            
            # Extract foreground embedding for S_NegLabel_star
            fg_embedding, cam_valid = self._extract_foreground_embedding(
                img, text_feature_c_hat
            )
            
            if not cam_valid:
                # Fallback to full image
                neg_sim_star = (100.0 * image_features @ self.neg_features_star.T)
                s_neglabel_star = float(self._neg_label_score_star(pos_sim, neg_sim_star).item())
                p_align = 0.0
            else:
                # Compute S_NegLabel_star on foreground
                # Add batch dimension for compatibility with scoring functions
                fg_pos_sim = (100.0 * fg_embedding @ self.pos_features.T).unsqueeze(0)  # (1, num_classes)
                fg_neg_sim = (100.0 * fg_embedding @ self.neg_features_star.T).unsqueeze(0)  # (1, num_neg)
                s_neglabel_star = float(self._neg_label_score_star(fg_pos_sim, fg_neg_sim).item())
                
                # Compute P_align on background (if enabled)
                if self.use_p_align:
                    p_align, _ = self._compute_p_align(img, orig_image, predicted_class_idx)
                else:
                    p_align = 0.0
        
        # Final score
        s_final = s_neglabel_star + self.lambda_bg * p_align
        
        if return_details:
            return {
                's_neglabel_star': s_neglabel_star,
                'p_align': p_align,
                'cam_valid': cam_valid,
                's_final': s_final,
                'predicted_class_idx': predicted_class_idx
            }
        else:
            return s_final
    
    def _extract_foreground_embedding(self, image_tensor, text_feature_c_hat):
        """
        Extract foreground embedding using CAM.
        
        Returns:
            fg_embedding: (D,) L2-normalized foreground embedding
            cam_valid: bool
        """
        try:
            # Compute CAM
            cam = self.cam_generator.compute_cam(image_tensor, text_feature_c_hat)
            
            # Get foreground mask (high CAM values)
            from cam_bg import cam_to_masks
            fg_mask, bg_mask = cam_to_masks(cam, self.cam_fg_percentile, self.cam_dilate_px)
            
            # Get patch tokens from CLIP visual encoder
            with torch.no_grad():
                model_dtype = next(self.clip_model.parameters()).dtype
                image_tensor_typed = image_tensor.to(model_dtype)
                
                # Extract patch embeddings
                x = self.clip_model.visual.conv1(image_tensor_typed)
                x = x.reshape(x.shape[0], x.shape[1], -1)
                x = x.permute(0, 2, 1)
                
                # Add class token
                class_token = self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                )
                x = torch.cat([class_token, x], dim=1)
                
                # Add positional embedding
                x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
                
                # Pass through transformer
                x = self.clip_model.visual.ln_pre(x)
                x = x.permute(1, 0, 2)
                x = self.clip_model.visual.transformer(x)
                x = x.permute(1, 0, 2)
                
                # Extract patch tokens
                patch_tokens = x[0, 1:, :]
            
            # Reshape foreground mask to match patch grid
            grid_size = int(np.sqrt(patch_tokens.shape[0]))
            fg_mask_resized = torch.nn.functional.interpolate(
                torch.from_numpy(fg_mask).unsqueeze(0).unsqueeze(0).float(),
                size=(grid_size, grid_size),
                mode='nearest'
            ).squeeze().numpy()
            
            fg_indices = fg_mask_resized.flatten() > 0
            
            if fg_indices.sum() == 0:
                return torch.zeros(patch_tokens.shape[1], device=patch_tokens.device), False
            
            # Pool foreground patches
            fg_patches = patch_tokens[fg_indices]
            fg_embedding = fg_patches.mean(dim=0)
            
            # Project through final layers
            fg_embedding = self.clip_model.visual.ln_post(fg_embedding)
            fg_embedding = fg_embedding @ self.clip_model.visual.proj
            
            # Normalize
            fg_embedding = fg_embedding / fg_embedding.norm()
            fg_embedding = fg_embedding.to(torch.float32)
            
            return fg_embedding, True
            
        except Exception as e:
            print(f"Warning: Foreground extraction failed: {e}")
            return torch.zeros(self.pos_features.shape[1], device=self.device), False
