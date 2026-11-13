"""HOUND CosPlace Visual Place Recognition Adapter.

Implements the CosPlace model (CVPR 2022) for scalable, geographic classification-based VPR.
Paper: "Rethinking Visual Geo-localization for Large-Scale Applications"
Authors: Gabriele Berton, Carlo Masone, Barbara Caputo (CVPR 2022)
License: MIT License
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from core.logger import get_logger
from core.vpr_base import BaseVPRModel, ModelProvenance, ModelUnavailableError, VPRExtractionError

logger = get_logger("hound.vpr.cosplace")


class CosPlaceModel(BaseVPRModel):
    """CosPlace deep metric learning place recognition model with GeM pooling."""

    def __init__(
        self,
        backbone: str = "ResNet18",
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
            model_name=f"CosPlace_{self.backbone_name}_{self.fc_output_dim}",
            paper_title="Rethinking Visual Geo-localization for Large-Scale Applications",
            paper_url="https://openaccess.thecvf.com/content/CVPR2022/papers/Berton_Rethinking_Visual_Geo-Localization_for_Large-Scale_Applications_CVPR_2022_paper.pdf",
            authors="Gabriele Berton, Carlo Masone, Barbara Caputo",
            year=2022,
            repository_url="https://github.com/gmberton/CosPlace",
            license_type="MIT License",
            training_dataset="SF-XL (San Francisco eXtra Large, 41.2M images)",
            weights_source="https://github.com/gmberton/CosPlace/releases/download/v1.0/ResNet18_512_cosplace.pth",
            descriptor_dim=self.fc_output_dim,
            default_input_resolution=(512, 512),
        )

    def _load_model(self) -> None:
        try:
            import torch
            import torchvision.transforms as transforms
        except ImportError as e:
            raise ModelUnavailableError(f"PyTorch or Torchvision not available for CosPlace: {e}") from e

        if self.use_cuda and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Load model using torch.hub or custom checkpoint
        try:
            if self.weights_path:
                logger.info(f"Loading CosPlace from local checkpoint: {self.weights_path}")
                self.model = torch.load(self.weights_path, map_location=self.device)
            else:
                logger.info(f"Loading CosPlace ({self.backbone_name}, dim={self.fc_output_dim}) via torch.hub...")
                self.model = torch.hub.load(
                    "gmberton/cosplace",
                    "get_trained_model",
                    backbone=self.backbone_name,
                    fc_output_dim=self.fc_output_dim,
                    trust_repo=True,
                )
            
            self.model.eval()
            self.model.to(self.device)

            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            logger.info(f"CosPlace loaded successfully on device '{self.device}'.")
        except Exception as e:
            self.model = None
            raise ModelUnavailableError(f"Failed to load CosPlace model weights: {e}") from e

    def _unload_model(self) -> None:
        self.model = None
        self.transform = None

    def _extract(self, image_rgb: np.ndarray) -> np.ndarray:
        if self.model is None or self.transform is None:
            raise VPRExtractionError("CosPlace model is not loaded.")

        import torch
        try:
            with torch.no_grad():
                tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
                features = self.model(tensor)
                vec = features.squeeze(0).cpu().numpy().astype(np.float32)
                return vec
        except Exception as e:
            raise VPRExtractionError(f"CosPlace extraction failed: {e}") from e
