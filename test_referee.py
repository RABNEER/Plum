import json
import shutil
import subprocess
import unittest
from pathlib import Path

import torch

from referee import CLAIM, HIDDEN, NOISE, ROOT, STALE, Referee, Window, event, generate


class RefereeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_revision_clears_only_its_requirement_after_long_failure_history(self):
        regression_fail = event(0, "strong_fail")
        regression_revision = event(0, "revision")
        build_pass = event(1, "strong_pass")
        tokens = torch.tensor([[build_pass] + [regression_fail] * 20 + [
            NOISE, STALE, CLAIM, regression_revision, regression_fail,
        ]])
        model = Referee().eval()
        model.load_state_dict(torch.load(ROOT / "referee.pt", weights_only=True)["sbm"])
        with torch.inference_mode():
            logits, state = model(tokens[:, :21])
            continuation, _ = model(tokens[:, 21:], state)
            readings = torch.cat((logits, continuation), dim=1).sigmoid()[0]
            from_new_revision = model(torch.tensor([[regression_fail]]))[0].sigmoid()[0, 0, 0]
        self.assertLess(readings[20, 0].item(), .25)
        self.assertEqual(readings[24, 0].item(), .5)
        torch.testing.assert_close(readings[25, 0], from_new_revision, rtol=0, atol=2e-6)
        torch.testing.assert_close(readings[:, 1:], readings[0, 1:].expand_as(readings[:, 1:]), rtol=0, atol=0)
        torch.testing.assert_close(readings[21:24], readings[20].expand_as(readings[21:24]), rtol=0, atol=0)

    def test_recorded_models_match_streaming_browser_inference(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required to verify browser inference parity")
        tokens, target, _, _, _ = generate(2, 192, .75, 72103)
        actual = subprocess.check_output(
            [node, "-e", """
                const fs = require("node:fs");
                global.window = {};
                require("./demo/weights.js");
                const BeliefSession = require("./demo/model.js");
                const tokens = JSON.parse(fs.readFileSync(0, "utf8"));
                const traces = tokens.map(stream => {
                    const session = new BeliefSession(window.SBM_WEIGHTS);
                    return stream.map(token => {
                        session.feed(token);
                        return session.query();
                    });
                });
                process.stdout.write(JSON.stringify(traces));
            """],
            cwd=ROOT, input=json.dumps(tokens.tolist()), text=True,
        )
        browser = json.loads(actual)
        weights = torch.load(Path(ROOT) / "referee.pt", weights_only=True)
        sbm = Referee().eval()
        baseline = Window().eval()
        sbm.load_state_dict(weights["sbm"])
        baseline.load_state_dict(weights["window"])
        with torch.inference_mode():
            logits, hidden = sbm(tokens)
            self.assertEqual(hidden.shape, (1, 2, HIDDEN))
            for name, expected in (("sbm", logits.sigmoid()),
                                   ("window", baseline(tokens).sigmoid()), ("oracle", target)):
                observed = torch.tensor([[step[name] for step in stream] for stream in browser])
                torch.testing.assert_close(observed, expected, rtol=0, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
