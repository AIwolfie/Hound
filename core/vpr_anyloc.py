"""HOUND AnyLoc Universal Visual Place Recognition Adapter.

Implements the AnyLoc foundation model approach (Keetha et al., RA-L 2023)
using self-supervised Vision Transformers (DINOv2) with GeM patch token aggregation.
Paper: "AnyLoc: Towards Universal Visual Place Recognition"
Authors: Nikhil Keetha, Avneesh Mishra, Jay Karhade, Kshitij Minhas, Sourav Garg, Sebastian Scherer
License: Apache 2.0
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from core.logger import get_logger
from core.vpr_base import BaseVPRModel, ModelProvenance, ModelUnavailableError, VPRExtractionError

logger = get_logger("hound.vpr.anyloc")


class AnyLocModel(BaseVPRModel):
    """AnyLoc zero-shot place recognition using DINOv2 self-supervised ViT representations."""

    def __init__(
        self,
        backbone: str = "dinov2_vits14",
        desc_dim: int = 384,
        use_cuda: bool = True,
        gem_p: float = 3.0,
    ):
        super().__init__(use_cuda=use_cuda)
        self.backbone_name = backbone
        self._desc_dim = desc_dim
        self.gem_p = gem_p
        self.model = None
        self.transform = None

    @property
    def provenance(self) -> ModelProvenance:
        return ModelProvenance(
            model_name=f"AnyLoc_{self.backbone_name}",
            paper_title="AnyLoc: Towards Universal Visual Place Recognition",
            paper_url="https://arxiv.org/abs/2308.00688",
            authors="Nikhil Keetha, Avneesh Mishra, Jay Karhade, Kshitij Minhas, Sourav Garg, Sebastian Scherer",
            year=2023,
            repository_url="https://github.com/AnyLoc/AnyLoc",
            license_type="Apache-2.0",
            training_dataset="LVD-142M (Self-supervised uncurated foundation dataset)",
            weights_source="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
            descriptor_dim=self._desc_dim,
            default_input_resolution=(448, 448),  # Multiple of 14 (14 * 32 = 448)
        )

    def _load_model(self) -> None:
        try:
            import torch
            import torchvision.transforms as transforms
        except ImportError as e:
            raise ModelUnavailableError(f"PyTorch or Torchvision not available for AnyLoc: {e}") from e

        if self.use_cuda and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        try:
            logger.info(f"Loading DINOv2 AnyLoc backbone '{self.backbone_name}' via torch.hub...")
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                self.backbone_name,
                trust_repo=True,
            )
            self.model.eval()
            self.model.to(self.device)

            # Input dimension must be multiple of patch size 14
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            logger.info(f"AnyLoc loaded successfully on device '{self.device}'.")
        except Exception as e:
            self.model = None
            raise ModelUnavailableError(f"Failed to load AnyLoc / DINOv2 model: {e}") from e

    def _unload_model(self) -> None:
        self.model = None
        self.transform = None

    def _extract(self, image_rgb: np.ndarray) -> np.ndarray:
        if self.model is None or self.transform is None:
            raise VPRExtractionError("AnyLoc model is not loaded.")

        import torch
        try:
            with torch.no_grad():
                tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
                
                # Extract features: DINOv2 provides forward_features with patch tokens
                if hasattr(self.model, "forward_features"):
                    features_dict = self.model.forward_features(tensor)
                    patch_tokens = features_dict.get("x_norm_patchtokens")  # shape: (1, num_patches, embed_dim)
                    if patch_tokens is not None:
                        # GeM pooling over patch tokens: (sum(x^p) / N)^(1/p)
                        x_clamped = torch.clamp(patch_tokens, min=1e-6)
                        gem = (torch.mean(x_clamped ** self.gem_p, dim=1)) ** (1.0 / self.gem_p)
                        vec = gem.squeeze(0).cpu().numpy().astype(np.float32)
                    else:
                        cls_token = features_dict.get("x_norm_clstoken")
                        vec = cls_token.squeeze(0).cpu().numpy().astype(np.float32)
                else:
                    # Direct forward returns CLS token
                    out = self.model(tensor)
                    vec = out.squeeze(0).cpu().numpy().astype(np.float32)

                return vec
        except Exception as e:
            raise VPRExtractionError(f"AnyLoc extraction failed: {e}") from e
