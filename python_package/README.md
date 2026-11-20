# Plum

Plum tracks structured check evidence across four channels. Inference runs offline using the Python standard library. The bundled Avadhanam-inspired GRU uses 660 parameters and carries 48 state values for the learned comparison.

## Quick start

Install Plum and try the built-in demo.

```sh
pip install ./python_package
plum demo
plum info
```

Run `plum` by itself to open the same demo. The two streams end with the same weak passing check. Plum gives different scores because the earlier full check failed in one stream and passed in the other.

```python
from plum_sbm import Plum

plum = Plum()
plum.feed("revision", "regression")
plum.feed("strong_fail", "regression", evidence_id="original-test")
for _ in range(10):
    plum.feed("noise")
plum.feed("claim")
print(f"{plum.query('regression'):.0%}")  # 10%

plum.feed("revision", "regression")
print(plum.query("regression"))          # 0.5
plum.feed("strong_pass", "regression", evidence_id="original-test")
print(f"{plum.query('regression'):.0%}")  # 90%
assert plum.ready("regression")
```

The four channels are `regression`, `build`, `integration`, and `style`. A `revision` clears the selected channel; `strong_pass`, `strong_fail`, `spot_pass`, and `spot_fail` update it. `noise`, `stale`, and `claim` leave the state unchanged. Use `plum.beliefs()` to read all four channels and `plum.reset()` to start a new session.

## Score a stream

Save one JSON object on each line of `events.jsonl`.

```jsonl
{"kind":"revision","channel":"regression"}
{"kind":"strong_fail","channel":"regression"}
{"kind":"strong_pass","channel":"build"}
{"kind":"claim"}
```

```sh
plum score events.jsonl
```

Each output line contains the input line number, updated probabilities, and a readiness flag for each channel. Check objects accept optional `evidence_id` and numeric `reliability` fields. Use `-` instead of a file path to read from standard input.

Use a stable `evidence_id` for multiple reports of the same observation. IDs are scoped to the requirement and cleared by a revision. Reports without IDs are counted as independent observations for scoring. Readiness requires an identified full pass, no full failure on the current revision, and a score of at least 85%. Conflicting reports for one ID block readiness for that revision and raise an error. The caller must identify the original full check and label out-of-date results as `stale`.

The evidence posterior assumes distinct checks are independent. Reliability defaults to .90 for a full check and .75 for a spot check. Supply a measured reliability from .50 up to but excluding 1 when available. Use `learned_query()` to read the original GRU score. Tracking IDs uses memory proportional to the identified checks on each current revision.

Plum is a research prototype for typed events. You supply the event tags. The model was trained on simulated checks with fixed evidence reliability, so treat its scores on other inputs as experimental. Run `plum info` for the bundled model card and benchmark results.
