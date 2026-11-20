# Plum — Independent Test Verdict (earlier checkpoint)

This report records the evaluation before Plum adopted separate recurrent states for each requirement. The current model routes evidence and resets by requirement. Calibration under different test reliability rates remains an open question. [Current model tests](test_referee.py) · [Published benchmark](referee-results.json)

**Tested:** 20 Nov 2026 · `referee.pt`, `demo/`, `referee.py`, `experiment.py`
**Method:** unit tests, checkpoint re-evaluation, fresh held-out streams, controlled behavior probes, and the static demo in real Chrome (desktop + 390px mobile).

## Verdict

**Pass for the synthetic benchmark and interactive demo. Not yet ready for real agent logs.**

The core claim holds: the model carries evidence across long distractor gaps and raises uncertainty when a completion claim contradicts an old failure. The published metrics are reproducible and the demo shows the intended behavior. Three structural weaknesses — revision reset, cross-channel leakage, and calibration under reliability shift — limit it beyond the simulator.

## Scorecard

| Area | Result |
|---|---|
| Unit tests | 4/4 pass |
| Standalone demo test | pass |
| Published metrics reproduced | yes, to 4 decimals |
| Browser ↔ PyTorch parity | within 2e-6 |
| Streaming equivalence | exact final hidden state |
| Real Chrome demo | no console errors; event 12 and 16 correct |
| Mobile 390px layout | fits, no page overflow |
| Fresh held-out seeds | beats window baseline at 75% and 90% distractors |
| Training scripts | execute and write outputs (2-step smoke test) |

## Key evidence

- **Published reproduction (192 events, 75% distractors):** Plum Brier 0.1216, window 0.2111, latest-check 0.1641, oracle 0.1178. Delayed MSE 0.0029 vs 0.1444.
- **Fresh streams:** 192/75% → 0.1167 vs 0.2105; 512/75% → 0.1046 vs 0.2107; 1024/90% → 0.1205 vs latest-check 0.1632 and oracle 0.1050.
- **Memory probe:** after a strong failure and 60 distractors, Plum 8.4% vs oracle 10% and window 48.4%.
- **Demo:** event 12 → 9% / 51% / 10% (Plum / baseline / oracle), verdict "Verification needed"; event 16 → 97% / 88% / 96%.

## Weaknesses

1. **Revision reset decays with history.** After 5 / 10 / 20 prior failures, a revision gives 22.6% / 11.4% / 8.2% instead of 50%. In-distribution, 26.7% of revisions follow 5+ checks, and 16.6% of all revisions are more than 10 points from 50%.
2. **Cross-channel leakage.** 20 strong failures on Regression move untouched channels to 71.4% / 37.9% / 39.0% instead of 50%. Reproduced in the live UI.
3. **Reliability shift.** At 70% / 60% check reliability instead of 90% / 75%, a strong pass reads 93.8% vs a true 70%; a spot pass reads 80.0% vs 60%. Brier 0.2022 vs shifted oracle 0.1864 (window 0.2443).

## Recommendation

**Ship the Devpost submission as scoped.** The README and SUBMISSION already describe the synthetic limits honestly. Before extending to real logs, add regression tests for the three probes and fix revision reset (explicit reset gate or per-channel state), channel isolation (per-channel recurrent state), and calibration (randomized reliability during training).

*This assessment covers an earlier checkpoint.*
