"""HOUND MixVPR Visual Place Recognition Adapter.

Implements the MixVPR architecture (Ali-bey et al., WACV 2023)
using spatial and channel feature mixing over multi-scale CNN feature maps.
Paper: "MixVPR: Feature Mixing for Visual Place Recognition"
Authors: Amar Ali-bey, Brahim Chaib-draa, Philippe Giguere (WACV 2023)
License: MIT License
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from core.logger import get_logger
from core.vpr_base import BaseVPRModel, ModelProvenance, ModelUnavailableError, VPRExtractionError

logger = get_logger("hound.vpr.mixvpr")


class MixVPRModel(BaseVPRModel):
    """MixVPR architecture combining deep CNN backbones with spatial-channel feature mixing."""

    def __init__(
        self,
        backbone: str = "resnet50",
        fc_output_dim: int = 512,
        use_cuda: bool = True,
        weights_path: Optional[str] = None,
    ):
        super().__init__(use_cuda=use_cuda)
        self.backbone_name = backbone
        self.fc_output_dim = fc_output_dim
        self.weights_path = weights_path
        self.model = None
        self.transform = None

    @property
    def provenance(self) -> ModelProvenance:
        return ModelProvenance(
            model_name=f"MixVPR_{self.backbone_name}_{self.fc_output_dim}",
            paper_title="MixVPR: Feature Mixing for Visual Place Recognition",
            paper_url="https://openaccess.thecvf.com/content/WACV2023/papers/Ali-Bey_MixVPR_Feature_Mixing_for_Visual_Place_Recognition_WACV_2023_paper.pdf",
            authors="Amar Ali-bey, Brahim Chaib-draa, Philippe Giguere",
            year=2023,
            repository_url="https://github.com/AmarAliBey/MixVPR",
            license_type="MIT License",
            training_dataset="GSV-Cities (560k images, 67k places)",
            weights_source="https://github.com/AmarAliBey/MixVPR",
            descriptor_dim=self.fc_output_dim,
            default_input_resolution=(320, 320),
        )

    def _load_model(self) -> None:
        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models
            import torchvision.transforms as transforms
        except ImportError as e:
            raise ModelUnavailableError(f"PyTorch or Torchvision not available for MixVPR: {e}") from e

        if self.use_cuda and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        try:
            # Build MixVPR PyTorch architecture
            class FeatureMixerBlock(nn.Module):
                def __init__(self, spatial_dim: int = 100, channels: int = 1024):
                    super().__init__()
                    self.spatial_mixer = nn.Sequential(
                        nn.Linear(spatial_dim, spatial_dim),
                        nn.GELU(),
                        nn.Linear(spatial_dim, spatial_dim),
                    )
                    self.channel_mixer = nn.Sequential(
                        nn.Linear(channels, channels),
                        nn.GELU(),
                        nn.Linear(channels, channels),
                    )

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # x shape: (B, C, HW)
                    # Mix along spatial dimension (HW)
                    x = x + self.spatial_mixer(x)
                    # Mix along channel dimension (C)
                    x_t = x.transpose(1, 2)  # (B, HW, C)
                    x_t = x_t + self.channel_mixer(x_t)
                    return x_t.transpose(1, 2)  # (B, C, HW)

            class MixVPRNetwork(nn.Module):
                def __init__(self, out_dim: int = 512):
                    super().__init__()
                    # ResNet-50 backbone up to layer3 (1024 channels)
                    base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                    self.backbone = nn.Sequential(
                        base.conv1, base.bn1, base.relu, base.maxpool,
                        base.layer1, base.layer2, base.layer3,
                    )
                    self.adaptive_pool = nn.AdaptiveAvgPool2d((10, 10))  # HW = 100
                    self.mixer1 = FeatureMixerBlock(spatial_dim=100, channels=1024)
                    self.mixer2 = FeatureMixerBlock(spatial_dim=100, channels=1024)
                    self.proj = nn.Linear(1024 * 100, out_dim)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    feats = self.backbone(x)  # (B, 1024, H, W)
                    pooled = self.adaptive_pool(feats)  # (B, 1024, 10, 10)
                    b, c, h, w = pooled.shape
                    flat_spatial = pooled.view(b, c, h * w)  # (B, 1024, 100)
                    mixed = self.mixer1(flat_spatial)
                    mixed = self.mixer2(mixed)
                    flat_all = mixed.view(b, -1)  # (B, 102400)
                    out = self.proj(flat_all)
                    return out

            self.model = MixVPRNetwork(out_dim=self.fc_output_dim)
            if self.weights_path:
                logger.info(f"Loading MixVPR custom weights from: {self.weights_path}")
                checkpoint = torch.load(self.weights_path, map_location=self.device)
                self.model.load_state_dict(checkpoint, strict=False)

            self.model.eval()
            self.model.to(self.device)

            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            logger.info(f"MixVPR loaded successfully on device '{self.device}'.")
        except Exception as e:
            self.model = None
            raise ModelUnavailableError(f"Failed to load MixVPR model: {e}") from e

    def _unload_model(self) -> None:
        self.model = None
        self.transform = None

    def _extract(self, image_rgb: np.ndarray) -> np.ndarray:
        if self.model is None or self.transform is None:
            raise VPRExtractionError("MixVPR model is not loaded.")

        import torch
        try:
            with torch.no_grad():
                tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
                feats = self.model(tensor)
                vec = feats.squeeze(0).cpu().numpy().astype(np.float32)
                return vec
        except Exception as e:
            raise VPRExtractionError(f"MixVPR extraction failed: {e}") from e
