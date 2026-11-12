"""Deterministic scientific validation tests for HOUND Solar Ephemeris & Feasibility."""
from __future__ import annotations

import math
import unittest
from datetime import datetime, timezone

from investigators.shadows.solar_analyzer import SolarAnalyzer, SolarPosition


class TestSolarValidation(unittest.TestCase):
    """Deterministic scientific test suite for NOAA solar calculations and feasibility ceilings."""

    def test_known_solar_positions_equinox_and_solstice(self):
        """Verify solar elevation and declination at known astronomical solstices and equinoxes."""
        # 1. March Equinox at the Equator (lat 0.0, lon 0.0, March 20 12:08 UTC)
        # Solar noon at prime meridian accounting for -7.4 min Equation of Time: sun reaches ~89.5°
        dt_equinox = datetime(2026, 3, 20, 12, 8, 0, tzinfo=timezone.utc)
        pos_eq = SolarAnalyzer.calculate_solar_position(0.0, 0.0, dt_equinox)
        self.assertAlmostEqual(pos_eq.elevation_deg, 89.54, delta=0.5)
        self.assertAlmostEqual(pos_eq.declination_deg, 0.0, delta=1.0)

        # 2. June Summer Solstice at Tropic of Cancer (lat 23.44°N, lon 0.0, June 21 12:00 UTC)
        # Sun passes directly through zenith (elevation ~90°, declination ~23.44°)
        dt_june_solstice = datetime(2026, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
        pos_june = SolarAnalyzer.calculate_solar_position(23.44, 0.0, dt_june_solstice)
        self.assertAlmostEqual(pos_june.elevation_deg, 90.0, delta=1.5)
        self.assertAlmostEqual(pos_june.declination_deg, 23.44, delta=0.5)

        # 3. December Winter Solstice at Tropic of Cancer (lat 23.44°N, lon 0.0, Dec 21 12:00 UTC)
        # Declination ~ -23.44°. Noon elevation = 90 - (23.44 - (-23.44)) = 43.12°
        dt_dec_solstice = datetime(2026, 12, 21, 12, 0, 0, tzinfo=timezone.utc)
        pos_dec = SolarAnalyzer.calculate_solar_position(23.44, 0.0, dt_dec_solstice)
        self.assertAlmostEqual(pos_dec.elevation_deg, 43.12, delta=1.5)
        self.assertAlmostEqual(pos_dec.declination_deg, -23.44, delta=0.5)

    def test_northern_and_southern_hemispheres(self):
        """Verify opposing seasons and sun azimuths between hemispheres."""
        # Dec 21 is summer solstice in the Southern Hemisphere
        dt_dec = datetime(2026, 12, 21, 2, 0, 0, tzinfo=timezone.utc)  # ~12:00 local time in Sydney (lon 151°)
        pos_sydney = SolarAnalyzer.calculate_solar_position(-33.86, 151.2, dt_dec)
        # In Southern Hemisphere summer, noon sun is high (>75°) and positioned to the NORTH (azimuth ~0° / 360°)
        self.assertGreater(pos_sydney.elevation_deg, 75.0)
        self.assertTrue(pos_sydney.azimuth_deg < 30.0 or pos_sydney.azimuth_deg > 330.0)

        # Dec 21 is winter solstice in the Northern Hemisphere (London, lat 51.5°N, lon 0.0)
        dt_london_noon = datetime(2026, 12, 21, 12, 0, 0, tzinfo=timezone.utc)
        pos_london = SolarAnalyzer.calculate_solar_position(51.5, 0.0, dt_london_noon)
        # Low noon sun (~15°) positioned to the SOUTH (azimuth ~180°)
        self.assertAlmostEqual(pos_london.elevation_deg, 15.1, delta=2.0)
        self.assertAlmostEqual(pos_london.azimuth_deg, 180.0, delta=10.0)

    def test_daylight_vs_nighttime(self):
        """Verify negative solar elevations at night and positive during daytime."""
        # London at midnight UTC
        dt_midnight = datetime(2026, 6, 21, 0, 0, 0, tzinfo=timezone.utc)
        pos_night = SolarAnalyzer.calculate_solar_position(51.5, 0.0, dt_midnight)
        self.assertLess(pos_night.elevation_deg, 0.0)
        self.assertIsNone(pos_night.shadow_length_ratio)

        # London at midday UTC
        dt_midday = datetime(2026, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
        pos_day = SolarAnalyzer.calculate_solar_position(51.5, 0.0, dt_midday)
        self.assertGreater(pos_day.elevation_deg, 50.0)
        self.assertIsNotNone(pos_day.shadow_length_ratio)

    def test_max_solar_elevation_physical_ceilings(self):
        """Verify rigorous physical solar ceiling formulas for various latitudes and seasons."""
        # Tropics: can reach 90° overhead
        self.assertEqual(SolarAnalyzer.compute_max_solar_elevation(0.0), 90.0)
        self.assertEqual(SolarAnalyzer.compute_max_solar_elevation(10.0), 90.0)
        self.assertEqual(SolarAnalyzer.compute_max_solar_elevation(23.44), 90.0)

        # Non-tropical latitudes (annual max = 113.44 - lat)
        self.assertAlmostEqual(SolarAnalyzer.compute_max_solar_elevation(34.0), 79.44, delta=0.1)
        self.assertAlmostEqual(SolarAnalyzer.compute_max_solar_elevation(45.0), 68.44, delta=0.1)
        self.assertAlmostEqual(SolarAnalyzer.compute_max_solar_elevation(65.0), 48.44, delta=0.1)
        self.assertAlmostEqual(SolarAnalyzer.compute_max_solar_elevation(90.0), 23.44, delta=0.1)

        # Specific day-of-year seasonal ceiling:
        # Paris (lat 48.85°N) on December 21 (day 355, decl ~ -23.44°)
        # Max noon elevation = 90 - (48.85 - (-23.44)) = 17.71°
        paris_winter_ceil = SolarAnalyzer.compute_max_solar_elevation(48.85, day_of_year=355)
        self.assertAlmostEqual(paris_winter_ceil, 17.71, delta=1.0)

    def test_solar_feasibility_falsification(self):
        """Verify active falsification of impossible solar observations."""
        # Case A: Iceland (lat 65.0°N) with observed sun elevation of 60°
        # Impossible at any time of year (annual ceiling is 48.44°)
        feasible, max_elev, reason = SolarAnalyzer.is_solar_elevation_feasible(65.0, 60.0)
        self.assertFalse(feasible)
        self.assertIn("exceeds maximum possible", reason)

        # Case B: Paris in December with observed sun elevation of 35°
        # Impossible in winter (December ceiling is ~17.7°)
        feasible_winter, _, _ = SolarAnalyzer.is_solar_elevation_feasible(48.85, 35.0, day_of_year=355)
        self.assertFalse(feasible_winter)

        # Case C: Tropics (10.0°N) with observed sun elevation of 85° in summer
        # Must be FEASIBLE (sun passes overhead in tropics)
        feasible_tropics, _, _ = SolarAnalyzer.is_solar_elevation_feasible(10.0, 85.0)
        self.assertTrue(feasible_tropics)

    def test_low_confidence_solar_shadow_safeguard(self):
        """Verify that tool registry does not falsify candidates when solar confidence is below 0.60."""
        from agent.tool_registry import ToolRegistry, ToolExecutionContext
        from storage.evidence_store import EvidenceItem, EvidenceStore
        from storage.database import DatabaseManager
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_solar.db"
            db = DatabaseManager(db_path)
            from storage.investigation_store import InvestigationStore
            from core.image_loader import load_and_assess_image
            import numpy as np
            import cv2

            dummy_img = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(dummy_img), np.zeros((100, 100, 3), dtype=np.uint8))
            img_rec = load_and_assess_image(dummy_img, preserve_original=False)
            inv_store = InvestigationStore(db, Path(tmpdir) / "storage")
            inv = inv_store.create_investigation(img_rec)

            ev_store = EvidenceStore(db)

            # Add low-confidence solar evidence (e.g. 0.35 confidence from noisy indoor/fuzzy shadow)
            ev_store.add_evidence(
                EvidenceItem(
                    investigation_id=inv.id,
                    kind="solar",
                    status="observed",
                    text="[Solar Ephemeris] Estimated solar elevation: 65.0°",
                    confidence_weight=0.35,  # Unreliable shadow measurement
                )
            )

            registry = ToolRegistry()
            ctx = ToolExecutionContext(
                investigation_id=inv.id,
                db_manager=db,
            )

            # Test against latitude 65.0°N (where 65° elevation is astronomically impossible)
            result = registry.execute_tool(
                "adversarial_solar_check",
                ctx,
                {"candidate_id": "cand_1", "lat": 65.0},
            )

            # Result must NOT be falsified because confidence was below 0.60 safeguard threshold
            self.assertEqual(result.status, "partial")
            self.assertEqual(result.output_data["status"], "untested")
            self.assertIn("below physical safety threshold", result.output_data["reason"])



if __name__ == "__main__":
    unittest.main()
