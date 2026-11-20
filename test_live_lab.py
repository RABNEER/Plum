"""Paired local test: the same recent evidence follows different full results."""

from pathlib import Path
import tempfile
import unittest

from live_lab import BACKGROUND, Lab
from referee import LOG_ODDS, KINDS, event, probability


class LiveEvidenceTest(unittest.TestCase):
    def test_same_ending_keeps_distinct_full_regression_evidence(self):
        spot_name = BACKGROUND[-1]
        latest_check_score = probability(LOG_ODDS[KINDS.index("spot_pass")])
        results = {}
        with tempfile.TemporaryDirectory(prefix="plum-paired-test-") as directory:
            for fixed in (False, True):
                folder = Path(directory) / ("fixed" if fixed else "broken")
                folder.mkdir()
                lab = Lab(folder)
                if fixed:
                    lab.patch()
                lab.check_regression()
                expected = "Passed" if fixed else "Failed"
                self.assertEqual(lab.status, expected)
                for name in BACKGROUND[:-1]:
                    lab.check_background(name)
                    self.assertTrue(lab.history[-1].label.endswith("pass"))

                # An order well above the threshold ships free in both cases.
                # This passing spot check cannot verify the $50 boundary.
                spot = lab.run_process(spot_name)
                self.assertEqual(spot.returncode, 0, spot.stderr)
                self.assertEqual(spot.stdout.strip(), f"{spot_name}: PASS")
                lab.feed(event(0, "spot_pass"), "Above-threshold spot check passed")
                lab.claim()
                results[fixed] = (list(lab.recent), lab.history[-1], lab.message)

        broken_tokens, broken, broken_verdict = results[False]
        fixed_tokens, fixed, fixed_verdict = results[True]
        self.assertEqual(broken_tokens, fixed_tokens)
        self.assertEqual(broken.window, fixed.window)
        self.assertLess(broken.plum, .40)
        self.assertGreaterEqual(fixed.plum, .85)
        self.assertIn("verify", broken_verdict)
        self.assertIn("supports", fixed_verdict)
        print("\nEarlier full regression   Plum    Last 8 events   Latest check")
        for name, reading in (("Failed", broken), ("Passed", fixed)):
            print(f"{name:<26} {reading.plum:>6.1%} {reading.window:>15.1%} {latest_check_score:>14.1%}")


if __name__ == "__main__":
    unittest.main()
