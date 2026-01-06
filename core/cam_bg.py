#!/usr/bin/env python3
"""
Grad-CAM for CLIP ViT to extract background embeddings.
NO SAM - pure gradient-based attention.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import binary_dilation


class ClipViTGradCAM:
    """
    Grad-CAM for CLIP ViT transformer blocks.
    Hooks into patch tokens to compute gradient-based attention maps.
    """

    def __init__(self, clip_model, target_layer_idx=-1, device='cuda:0'):
        """
        Args:
            clip_model: CLIP model instance
            target_layer_idx: which transformer block to hook (-1 = last block)
            device: CUDA device
        """
        self.clip_model = clip_model
        self.device = device

        # Access the vision transformer
        self.visual = clip_model.visual

        # Hook into transformer block
        self.target_layer = self.visual.transformer.resblocks[target_layer_idx]

        # Storage for activations and gradients
        self.activations = None
        self.gradients = None

        # Register hooks
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """Store patch token features during forward pass."""
        # output shape: (batch, seq_len, dim) or (seq_len, batch, dim) depending on CLIP version
        # seq_len = 1 (CLS) + num_patches
        if isinstance(output, tuple):
            output = output[0]
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Store gradients during backward pass."""
        # grad_output[0] shape: (batch, seq_len, dim)
        self.gradients = grad_output[0].detach()

    def __del__(self):
        """Clean up hooks."""
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()

    def compute_cam(self, image_tensor, text_feature):
        """
        Compute CAM heatmap for given image and target text feature.

        Args:
            image_tensor: (1, 3, H, W) preprocessed CLIP image
            text_feature: (D,) target text embedding (e.g., predicted class)

        Returns:
            cam: (H_patches, W_patches) normalized heatmap in [0, 1]
        """
        # Match dtype with CLIP model
        model_dtype = next(self.clip_model.parameters()).dtype
        
        # Convert to model dtype (creates new tensor)
        image_tensor = image_tensor.to(model_dtype)
        
        # Enable gradient computation on the new tensor
        image_tensor = image_tensor.requires_grad_(True)

        # Forward pass - MUST enable gradients even in eval mode
        with torch.set_grad_enabled(True):
            image_features = self.clip_model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute similarity score (target for gradient)
            # Ensure dtype matches
            text_feature = text_feature.to(image_features.dtype)
            text_feature_normalized = text_feature / text_feature.norm()
            similarity = (image_features @ text_feature_normalized.unsqueeze(1)).squeeze()

            # Backward pass
            self.clip_model.zero_grad()
            similarity.backward()

        # Get activations and gradients
        # CLIP uses (seq_len, batch, dim) format in transformer
        if self.activations is None:
            raise RuntimeError("No activations captured. Check hook registration.")
        if self.gradients is None:
            raise RuntimeError("No gradients captured. Check backward hook.")

        # Handle both (seq_len, batch, dim) and (batch, seq_len, dim)
        if self.activations.ndim == 3:
            if self.activations.shape[1] == 1:  # (seq_len, 1, dim)
                activations = self.activations[:, 0, :]  # (seq_len, dim)
                gradients = self.gradients[:, 0, :]  # (seq_len, dim)
            else:  # (1, seq_len, dim)
                activations = self.activations[0]  # (seq_len, dim)
                gradients = self.gradients[0]  # (seq_len, dim)
        else:
            activations = self.activations
            gradients = self.gradients

        # Remove CLS token (first token)
        patch_activations = activations[1:]  # (num_patches, dim)
        patch_gradients = gradients[1:]  # (num_patches, dim)

        # Compute channel-wise importance (alpha_k)
        weights = patch_gradients.mean(dim=0)  # (dim,)

        # Weighted combination of feature maps
        cam = (patch_activations * weights.unsqueeze(0)).sum(dim=1)  # (num_patches,)

        # Apply ReLU
        cam = F.relu(cam)

        # Reshape to spatial grid
        num_patches = patch_activations.shape[0]
        grid_size = int(np.sqrt(num_patches))
        assert grid_size * grid_size == num_patches, f"num_patches={num_patches} is not a perfect square"

        cam = cam.view(grid_size, grid_size)

        # Normalize to [0, 1]
        cam_min = cam.min() if cam.numel() > 0 else 0.0
        cam_max = cam.max() if cam.numel() > 0 else 1.0
        cam = cam - cam_min
        if cam_max - cam_min > 0:
            cam = cam / (cam_max - cam_min)

        # Clear gradients
        image_tensor.requires_grad = False

        return cam.cpu().numpy()


def cam_to_masks(cam, fg_percentile=80, dilate_px=1):
    """
    Convert CAM heatmap to foreground/background masks.

    Args:
        cam: (H, W) numpy array in [0, 1]
        fg_percentile: percentile threshold for foreground (default: 80)
        dilate_px: dilation pixels to reduce object-edge leakage (default: 1)

    Returns:
        fg_mask: (H, W) bool array
        bg_mask: (H, W) bool array
    """
    # Compute threshold
    threshold = np.percentile(cam, fg_percentile)

    # Foreground mask
    fg_mask = cam >= threshold

    # Dilate foreground to reduce edge leakage
    if dilate_px > 0:
        fg_mask = binary_dilation(fg_mask, iterations=dilate_px)

    # Background mask
    bg_mask = ~fg_mask

    # Ensure masks are non-empty (fallback)
    if not bg_mask.any():
        # If no background, use bottom 20% of cam
        threshold_bg = np.percentile(cam, 20)
        bg_mask = cam <= threshold_bg

    if not fg_mask.any():
        # If no foreground, use top 20% of cam
        threshold_fg = np.percentile(cam, 80)
        fg_mask = cam >= threshold_fg

    return fg_mask, bg_mask


def masked_pooling(patch_tokens, mask, patch_grid_size, mode='mean'):
    """
    Pool patch tokens using spatial mask.

    Args:
        patch_tokens: (num_patches, dim) tensor
        mask: (H_patches, W_patches) bool array
        patch_grid_size: int, grid size (e.g., 7 for 224px image with patch_size=32)
        mode: 'mean' or 'weighted' (default: 'mean')

    Returns:
        pooled: (dim,) tensor, L2-normalized
    """
    # Flatten mask to match patch tokens
    mask_flat = torch.from_numpy(mask.flatten()).to(patch_tokens.device)

    # Select patches where mask=True
    selected_patches = patch_tokens[mask_flat]  # (num_selected, dim)

    if selected_patches.shape[0] == 0:
        # Fallback: return mean of all patches
        pooled = patch_tokens.mean(dim=0)
    else:
        if mode == 'mean':
            pooled = selected_patches.mean(dim=0)
        elif mode == 'weighted':
            # Weighted by mask values (not implemented yet, fallback to mean)
            pooled = selected_patches.mean(dim=0)
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")

    # L2 normalize
    pooled = pooled / pooled.norm()

    return pooled


def extract_bg_embedding(clip_model, image_tensor, text_feature_for_c_hat,
                         cam_generator, fg_percentile=80, dilate_px=1):
    """
    Extract background embedding using Grad-CAM.

    Args:
        clip_model: CLIP model
        image_tensor: (1, 3, H, W) preprocessed image
        text_feature_for_c_hat: (D,) text feature for predicted class
        cam_generator: ClipViTGradCAM instance
        fg_percentile: foreground percentile threshold
        dilate_px: dilation pixels

    Returns:
        bg_embedding: (D,) L2-normalized background embedding
        cam_valid: bool, True if CAM was successfully computed
    """
    try:
        # Compute CAM
        cam = cam_generator.compute_cam(image_tensor, text_feature_for_c_hat)

        # Get masks
        fg_mask, bg_mask = cam_to_masks(cam, fg_percentile, dilate_px)

        # Get patch tokens from CLIP visual encoder
        with torch.no_grad():
            # Ensure dtype matches model
            model_dtype = next(clip_model.parameters()).dtype
            image_tensor_typed = image_tensor.to(model_dtype)

            # Extract patch embeddings (before final projection)
            x = clip_model.visual.conv1(image_tensor_typed)  # (1, width, grid, grid)
            x = x.reshape(x.shape[0], x.shape[1], -1)  # (1, width, grid**2)
            x = x.permute(0, 2, 1)  # (1, grid**2, width)

            # Add class token
            class_token = clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            )
            x = torch.cat([class_token, x], dim=1)  # (1, grid**2+1, width)

            # Add positional embedding
            x = x + clip_model.visual.positional_embedding.to(x.dtype)

            # Pass through transformer
            x = clip_model.visual.ln_pre(x)
            x = x.permute(1, 0, 2)  # (seq_len, 1, width)
            x = clip_model.visual.transformer(x)
            x = x.permute(1, 0, 2)  # (1, seq_len, width)

            # Extract patch tokens (remove CLS token)
            patch_tokens = x[0, 1:, :]  # (grid**2, width)

        # Get grid size
        grid_size = int(np.sqrt(patch_tokens.shape[0]))

        # Pool background patches (in transformer space, before projection)
        bg_embedding_pre = masked_pooling(patch_tokens, bg_mask, grid_size, mode='mean')  # (width,)

        # Project to CLIP embedding space (width -> embed_dim)
        # CLIP ViT applies ln_post and proj to CLS token, we apply to pooled bg patches
        if hasattr(clip_model.visual, 'proj') and clip_model.visual.proj is not None:
            bg_embedding = bg_embedding_pre @ clip_model.visual.proj  # (width,) @ (width, embed_dim) = (embed_dim,)
            bg_embedding = bg_embedding / bg_embedding.norm()
        else:
            bg_embedding = bg_embedding_pre

        return bg_embedding, True

    except Exception as e:
        print(f"Warning: CAM extraction failed: {e}")
        # Return zero embedding
        return torch.zeros(text_feature_for_c_hat.shape[0], device=text_feature_for_c_hat.device), False


# Sanity check function
def test_cam_extraction():
    """Test CAM extraction with a dummy image."""
    import clip
    from PIL import Image

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load CLIP
    clip_model, preprocess = clip.load('ViT-B/16', device=device)
    clip_model.eval()
    clip_model.float()  # Convert to float32 for gradients

    # Create CAM generator
    cam_gen = ClipViTGradCAM(clip_model, target_layer_idx=-1, device=device)

    # Load test image (use a solid color for testing)
    test_image = Image.new('RGB', (224, 224), color='red')
    image_tensor = preprocess(test_image).unsqueeze(0).to(device)

    # Create dummy text feature
    text = clip.tokenize(["a photo of a dog"]).to(device)
    with torch.no_grad():
        text_feature = clip_model.encode_text(text)[0]

    # Extract background embedding
    bg_emb, valid = extract_bg_embedding(
        clip_model, image_tensor, text_feature, cam_gen,
        fg_percentile=80, dilate_px=1
    )

    print(f"Background embedding shape: {bg_emb.shape}")
    print(f"Background embedding norm: {bg_emb.norm().item():.4f}")
    print(f"CAM valid: {valid}")

    return bg_emb, valid


if __name__ == '__main__':
    print("Testing CAM extraction...")
    bg_emb, valid = test_cam_extraction()
    print("Test completed successfully!")
