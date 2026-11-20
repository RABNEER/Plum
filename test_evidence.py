"""Regression tests for evidence and readiness on difficult streams."""

from itertools import combinations
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "python_package" / "src"))
from plum_sbm import Plum


class EvidenceTests(unittest.TestCase):
    def test_click_retries_count_once(self):
        plum = Plum()
        plum.feed("strong_fail", "regression", evidence_id="original")
        for _ in range(8):
            plum.feed("spot_pass", "regression", evidence_id="neighbor")
        self.assertAlmostEqual(plum.query("regression"), .25)
        self.assertFalse(plum.ready("regression"))

    def test_proxy_passes_cannot_verify_original(self):
        plum = Plum()
        for check in ("one", "two"):
            plum.feed("spot_pass", "regression", evidence_id=check)
        self.assertAlmostEqual(plum.query("regression"), .90)
        self.assertFalse(plum.ready("regression"))
        plum.feed("strong_pass", "regression")
        self.assertFalse(plum.ready("regression"))
        plum.feed("strong_pass", "regression", evidence_id="original")
        self.assertTrue(plum.ready("regression"))

    def test_balanced_checks_end_at_half_in_every_order(self):
        for passes in combinations(range(12), 6):
            plum = Plum()
            for index in range(12):
                kind = "strong_pass" if index in passes else "strong_fail"
                plum.feed(kind, "regression", evidence_id=str(index))
            self.assertEqual(plum.query("regression"), .5)
            self.assertFalse(plum.ready("regression"))

    def test_conflict_blocks_until_revision_and_preserves_other_channels(self):
        plum = Plum()
        for channel in ("build", "regression"):
            plum.feed("strong_pass", channel, evidence_id="full")
            self.assertTrue(plum.ready(channel))
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            plum.feed("strong_fail", "regression", evidence_id="full")
        self.assertFalse(plum.ready("regression"))
        plum.feed("revision", "regression")
        self.assertEqual(plum.query("regression"), .5)
        self.assertTrue(plum.ready("build"))
        plum.feed("strong_pass", "regression", evidence_id="full")
        self.assertTrue(plum.ready("regression"))

    def test_supplied_reliability_controls_the_posterior(self):
        plum = Plum()
        plum.feed("strong_pass", "regression", evidence_id="full", reliability=.70)
        self.assertAlmostEqual(plum.query("regression"), .70)
        self.assertFalse(plum.ready("regression"))
        plum.feed("spot_fail", "regression", evidence_id="spot", reliability=.60)
        self.assertAlmostEqual(plum.query("regression"), 14 / 23)
        for rate in (True, 1, .49, float("nan"), "0.7"):
            with self.assertRaises(ValueError):
                plum.feed("strong_pass", "regression", reliability=rate)


if __name__ == "__main__":
    unittest.main()
