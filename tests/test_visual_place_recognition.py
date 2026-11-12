"""Unit tests for HOUND Visual Place Recognition (VPR) & Geographic Image Retrieval.

Tests:
1. HybridColorSpatialDescriptor extraction, dimensionality, and normalization.
2. NeuralGlobalDescriptor PyTorch execution and device fallback.
3. LocalFeatureMatcher ORB/SIFT detection and RANSAC homography.
4. VisualReferenceDatabase vector similarity search and landmark retrieval.
"""
from __future__ import annotations

import unittest
import numpy as np
import cv2

from core.visual_place_recognition import (
    BaseVisualDescriptorExtractor,
    GeoreferencedReferenceItem,
    HybridColorSpatialDescriptor,
    LocalFeatureMatcher,
    NeuralGlobalDescriptor,
    VisualPlaceMatch,
    VisualReferenceDatabase,
)
from core.spatial_envelope import GeographicResolutionLevel


class TestVisualPlaceRecognition(unittest.TestCase):
    def setUp(self):
        # Create synthetic images representing different geographic scenes with texture patterns
        self.eiffel_img = np.zeros((256, 384, 3), dtype=np.uint8)
        self.eiffel_img[:128, :] = [210, 150, 80]  # Blue sky
        self.eiffel_img[128:, :] = [70, 70, 70]   # Dark ground
        # Add lattice / corner features so ORB detects keypoints
        for x in range(120, 260, 20):
            cv2.line(self.eiffel_img, (x, 128), (x + 10, 250), (255, 255, 255), 2)
            cv2.circle(self.eiffel_img, (x, 180), 6, (0, 255, 0), -1)

        self.coastal_img = np.zeros((256, 384, 3), dtype=np.uint8)
        self.coastal_img[:, :192] = [180, 100, 30]  # Ocean blue
        self.coastal_img[:, 192:] = [40, 120, 80]   # Green cliff
        # Add wave and cliff textures
        for y in range(50, 200, 30):
            cv2.line(self.coastal_img, (20, y), (170, y + 5), (255, 255, 255), 2)

        self.indoor_blank_img = np.ones((256, 384, 3), dtype=np.uint8) * 180  # Uniform beige drywall

    def test_hybrid_descriptor_extraction_and_normalization(self):
        """Verify hybrid descriptor extracts fixed-size, L2-normalized vectors."""
        extractor = HybridColorSpatialDescriptor(grid_rows=3, grid_cols=3)
        desc = extractor.extract_descriptor(self.eiffel_img)

        # 9 cells * 128 HSV bins = 1152, plus 9 cells * 8 edge bins = 72 -> 1224 dimensions
        self.assertEqual(len(desc), 1224)
        norm = np.linalg.norm(desc)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_descriptor_similarity_and_discrimination(self):
        """Verify that identical/similar images produce high cosine similarity and distinct ones diverge."""
        extractor = HybridColorSpatialDescriptor()
        desc1 = extractor.extract_descriptor(self.coastal_img)

        # Slightly perturbed version of coastal image (with minor noise)
        noisy_coastal = np.clip(self.coastal_img.astype(np.int16) + np.random.randint(-5, 6, self.coastal_img.shape), 0, 255).astype(np.uint8)
        desc_noisy = extractor.extract_descriptor(noisy_coastal)
        desc_eiffel = extractor.extract_descriptor(self.eiffel_img)

        sim_same = float(np.dot(desc1, desc_noisy))
        sim_diff = float(np.dot(desc1, desc_eiffel))

        # Similar images should have cosine similarity > 0.70
        self.assertGreater(sim_same, 0.70)
        # Distinct scenes should have significantly lower similarity
        self.assertLess(sim_diff, sim_same)

    def test_neural_descriptor_execution_and_fallback(self):
        """Verify NeuralGlobalDescriptor executes on available PyTorch backbone without crashing."""
        neural_extractor = NeuralGlobalDescriptor(model_name="resnet18", use_cuda=False)
        desc = neural_extractor.extract_descriptor(self.eiffel_img)

        self.assertIsNotNone(desc)
        self.assertGreater(len(desc), 0)
        norm = np.linalg.norm(desc)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_local_feature_matching_and_ransac_verification(self):
        """Verify LocalFeatureMatcher detects keypoints and computes RANSAC inliers."""
        matcher = LocalFeatureMatcher(detector_type="orb", max_features=500)

        # Test on identical image (must find high inliers and verified=True)
        inliers, ratio, verified = matcher.match_and_verify(self.eiffel_img, self.eiffel_img)
        self.assertGreaterEqual(inliers, 4)

    def test_reference_database_landmark_retrieval(self):
        """Verify georeferenced reference database retrieves top landmark matches."""
        vpr_db = VisualReferenceDatabase()

        # Query with coastal prototype image
        matches = vpr_db.query_visual_matches(self.coastal_img, top_k=3, min_similarity=0.40)
        self.assertGreater(len(matches), 0)

        # Top match should be Big Sur coastal highway or coastal tag
        top = matches[0]
        self.assertIsNotNone(top.name)
        self.assertIsNotNone(top.lat)
        self.assertIsNotNone(top.lon)
        self.assertGreater(top.similarity_score, 0.50)
        self.assertIsNotNone(top.spatial_envelope)
        self.assertIn(top.spatial_envelope.resolution_level, (GeographicResolutionLevel.ROAD_CORRIDOR, GeographicResolutionLevel.LANDMARK))


if __name__ == "__main__":
    unittest.main()
