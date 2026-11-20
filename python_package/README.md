# Plum

Plum is an Avadhanam-inspired stateful belief model. It reads structured evidence as it arrives and estimates whether each requirement will pass. The trained model uses 660 parameters and carries 48 values across four independent channels. Inference runs offline using the Python standard library.

## Quick start

Install Plum and try the built-in demo.

```sh
pip install plum-sbm
plum demo
plum info
```

Run `plum` by itself to open the same demo. The two streams end with the same weak passing check. Plum gives different scores because the earlier full check failed in one stream and passed in the other.

```python
from plum_sbm import Plum

plum = Plum()
plum.feed("revision", "regression")
plum.feed("strong_fail", "regression")
for _ in range(10):
    plum.feed("noise")
plum.feed("claim")
print(f"{plum.query('regression'):.0%}")  # 10%

plum.feed("revision", "regression")
print(plum.query("regression"))          # 0.5
plum.feed("strong_pass", "regression")
print(f"{plum.query('regression'):.0%}")  # 90%
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

Each output line contains the input line number and the updated probability for each channel. Use `-` instead of a file path to read from standard input.

Plum is a research prototype for typed events. You supply the event tags. The model was trained on simulated checks with fixed evidence reliability, so treat its scores on other inputs as experimental. Run `plum info` for the bundled model card and benchmark results.
