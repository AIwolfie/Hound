"""HOUND Legacy & Baseline Visual Place Recognition Adapters.

Wraps existing HOUND descriptors (Hybrid Color-Spatial and ImageNet ResNet)
into the unified BaseVPRModel contract for rigorous backward compatibility.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.logger import get_logger
from core.vpr_base import BaseVPRModel, ModelProvenance, VPRExtractionError

logger = get_logger("hound.vpr.legacy")


class HybridColorSpatialModel(BaseVPRModel):
    """Handcrafted 1224-d color pyramid and steerable edge gradient descriptor."""

    def __init__(self, grid_rows: int = 3, grid_cols: int = 3):
        super().__init__(use_cuda=False)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.device = "cpu"

    @property
    def provenance(self) -> ModelProvenance:
        return ModelProvenance(
            model_name="Hybrid_Color_Spatial_1224",
            paper_title="HOUND Spatial Color Pyramid & Steerable Edge Descriptor",
            paper_url="docs/HOUND_ARCHITECTURE.md",
            authors="HOUND Core Team",
            year=2026,
            repository_url="https://github.com/AIwolfie/Hound",
            license_type="MIT License",
            training_dataset="None (Handcrafted Algorithmic Filter)",
            weights_source="None",
            descriptor_dim=1224,
            default_input_resolution=(256, 384),
        )

    def _load_model(self) -> None:
        self.device = "cpu"

    def _unload_model(self) -> None:
        pass

    def _extract(self, image_rgb: np.ndarray) -> np.ndarray:
        from core.visual_place_recognition import HybridColorSpatialDescriptor
        extractor = HybridColorSpatialDescriptor(grid_rows=self.grid_rows, grid_cols=self.grid_cols)
        return extractor.extract_descriptor(image_rgb)


class BaselineNeuralModel(BaseVPRModel):
    """Baseline deep CNN descriptor using ImageNet-pretrained ResNet-18."""

    def __init__(self, use_cuda: bool = True):
        super().__init__(use_cuda=use_cuda)
        self.model = None
        self.transform = None

    @property
    def provenance(self) -> ModelProvenance:
        return ModelProvenance(
            model_name="ResNet18_ImageNet_Baseline",
            paper_title="Deep Residual Learning for Image Recognition",
            paper_url="https://arxiv.org/abs/1512.03385",
            authors="Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun",
            year=2015,
            repository_url="https://github.com/pytorch/vision",
            license_type="BSD-3-Clause",
            training_dataset="ImageNet-1k (Object Classification)",
            weights_source="torchvision.models.ResNet18_Weights.DEFAULT",
            descriptor_dim=512,
            default_input_resolution=(224, 224),
        )

    def _load_model(self) -> None:
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms

        self.device = "cuda" if (self.use_cuda and torch.cuda.is_available()) else "cpu"
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _unload_model(self) -> None:
        self.model = None
        self.transform = None

    def _extract(self, image_rgb: np.ndarray) -> np.ndarray:
        import torch
        if self.model is None or self.transform is None:
            raise VPRExtractionError("Baseline neural model is not loaded.")
        with torch.no_grad():
            tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            feats = self.model(tensor)
            return feats.squeeze(0).cpu().numpy().astype(np.float32)
