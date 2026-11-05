"""Unit tests for HOUND Geographic Candidate Comparator."""
from __future__ import annotations

import unittest

from agent.candidate_comparator import CandidateComparator
from agent.candidate_generator import GeographicCandidate


class TestCandidateComparator(unittest.TestCase):
    """Test suite for CandidateComparator scoring, weighting, and ranking."""

    def test_level_weighting_hierarchy(self):
        """Verify more specific candidates receive higher level multipliers."""
        c_point = GeographicCandidate(
            id="C1",
            investigation_id="inv1",
            level="point",
            name="Specific GPS Spot",
            score=2.0,
            status="active",
        )
        c_region = GeographicCandidate(
            id="C2",
            investigation_id="inv1",
            level="region",
            name="General Country/Region",
            score=2.0,
            status="active",
        )

        ranked = CandidateComparator.compare_and_rank([c_region, c_point])
        self.assertEqual(len(ranked), 2)
        # Point level multiplier is 5.0 -> 2.0 * 5.0 = 10.0
        # Region level multiplier is 1.0 -> 2.0 * 1.0 = 2.0
        self.assertEqual(ranked[0].candidate_id, "C1")
        self.assertEqual(ranked[0].rank, 1)
        self.assertAlmostEqual(ranked[0].composite_score, 10.0)
        self.assertEqual(ranked[1].candidate_id, "C2")
        self.assertEqual(ranked[1].rank, 2)
        self.assertAlmostEqual(ranked[1].composite_score, 2.0)

    def test_terrain_correlation_boost(self):
        """Verify candidates with matching terrain horizons receive a score bonus."""
        c1 = GeographicCandidate(
            id="C1",
            investigation_id="inv1",
            level="area",
            name="Alpine Valley",
            score=3.0,
            status="active",
        )
        c2 = GeographicCandidate(
            id="C2",
            investigation_id="inv1",
            level="area",
            name="Flat Plain",
            score=3.0,
            status="active",
        )

        terrain_scores = {"C1": 0.85, "C2": 0.0}
        ranked = CandidateComparator.compare_and_rank([c2, c1], terrain_scores=terrain_scores)

        # C1 gets terrain multiplier: 1.0 + (0.85 * 2.0) = 2.7
        # composite_score: 3.0 * 2.5 (area) * 2.7 = 20.25
        self.assertEqual(ranked[0].candidate_id, "C1")
        self.assertGreater(ranked[0].composite_score, ranked[1].composite_score)
        self.assertTrue(any("DEM horizon correlation" in r for r in ranked[0].supporting_reasons))

    def test_contradiction_severe_penalty(self):
        """Verify contradiction severely penalizes score and demotes candidate."""
        c_active = GeographicCandidate(
            id="C1",
            investigation_id="inv1",
            level="city",
            name="Valid City",
            score=2.0,
            status="active",
        )
        c_contradicted = GeographicCandidate(
            id="C2",
            investigation_id="inv1",
            level="point",
            name="Falsified Point",
            score=10.0,
            status="active",
            contradictory_evidence_ids=["EV_SOLAR_FAIL"],
            falsification_condition="Solar elevation impossible.",
        )

        ranked = CandidateComparator.compare_and_rank([c_contradicted, c_active])

        # Even though C2 has higher base score and point level (50.0),
        # 1 contradiction penalizes it by 0.10x -> 5.0 and marks it excluded.
        self.assertEqual(ranked[0].candidate_id, "C1")
        self.assertEqual(ranked[0].status, "active")

        c2_eval = next(r for r in ranked if r.candidate_id == "C2")
        self.assertEqual(c2_eval.status, "excluded")
        self.assertTrue(any("Solar elevation impossible" in r for r in c2_eval.contradictory_reasons))


if __name__ == "__main__":
    unittest.main()
