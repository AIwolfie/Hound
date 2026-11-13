"""HOUND Phase 7 VPR Model Adapters Test Suite.

Verifies:
- Documented model provenance, citation, and licensing.
- Lazy loading and memory unloading.
- Strict error handling without silent substitution.
- Exact L2 normalization and expected descriptor dimensionality.
"""
import numpy as np
import pytest

from core.vpr_anyloc import AnyLocModel
from core.vpr_base import BaseVPRModel, ModelProvenance, ModelUnavailableError, VPRExtractionError
from core.vpr_cosplace import CosPlaceModel
from core.vpr_legacy import BaselineNeuralModel, HybridColorSpatialModel
from core.vpr_mixvpr import MixVPRModel


def test_cosplace_provenance_and_contract():
    model = CosPlaceModel(backbone="ResNet18", fc_output_dim=512, use_cuda=False)
    prov = model.provenance
    assert isinstance(prov, ModelProvenance)
    assert prov.model_name == "CosPlace_ResNet18_512"
    assert "Berton" in prov.authors
    assert prov.year == 2022
    assert prov.license_type == "MIT License"
    assert prov.descriptor_dim == 512
    assert model.is_loaded is False


def test_anyloc_provenance_and_contract():
    model = AnyLocModel(backbone="dinov2_vits14", desc_dim=384, use_cuda=False)
    prov = model.provenance
    assert prov.model_name == "AnyLoc_dinov2_vits14"
    assert "Keetha" in prov.authors
    assert prov.license_type == "Apache-2.0"
    assert prov.descriptor_dim == 384
    assert model.is_loaded is False


def test_mixvpr_provenance_and_contract():
    model = MixVPRModel(backbone="resnet50", fc_output_dim=512, use_cuda=False)
    prov = model.provenance
    assert prov.model_name == "MixVPR_resnet50_512"
    assert "Ali-bey" in prov.authors
    assert prov.license_type == "MIT License"
    assert prov.descriptor_dim == 512
    assert model.is_loaded is False


def test_cosplace_lazy_loading_and_normalization():
    model = CosPlaceModel(backbone="ResNet18", fc_output_dim=512, use_cuda=False)
    assert model.is_loaded is False

    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    vec = model.extract_descriptor(dummy_img)

    assert model.is_loaded is True
    assert vec.shape == (512,)
    assert pytest.approx(np.linalg.norm(vec), abs=1e-5) == 1.0
    assert model.last_telemetry is not None
    assert model.last_telemetry.is_l2_normalized is True

    model.unload()
    assert model.is_loaded is False


def test_zero_silent_substitution_on_invalid_input():
    model = CosPlaceModel(backbone="ResNet18", fc_output_dim=512, use_cuda=False)
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)

    with pytest.raises(VPRExtractionError):
        model.extract_descriptor(empty_img)


def test_hybrid_legacy_adapter_contract():
    model = HybridColorSpatialModel()
    assert model.descriptor_dim == 1224
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    vec = model.extract_descriptor(img)
    assert vec.shape == (1224,)
    assert pytest.approx(np.linalg.norm(vec), abs=1e-5) == 1.0
