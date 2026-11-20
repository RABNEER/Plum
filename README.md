# Plum · The Belief Lab

A coding agent can finish with green checks after an earlier regression has slipped out of view. Plum is a **660-parameter stateful model** that reads typed events once and estimates whether each original requirement passes.

With the same passing checks at the end of both runs, Plum gave Click’s original regression test a **24.8%** chance of passing before the upstream fix and **96.5%** after. [Test output](submission/Plum-Click-Test.txt)

**[Try the live demo](https://plum-sbm-demo.vercel.app/)** · [PyTorch model](referee.py) · [Method and reproduction](METHOD.md)

## Try it

The Python package includes the trained weights and runs offline on Python 3.10 or newer.

```sh
pip install plum-sbm
plum demo
```

```python
from plum_sbm import Plum

plum = Plum()
plum.feed("revision", "regression")
plum.feed("strong_fail", "regression")
for _ in range(10):
    plum.feed("noise")
print(f"{plum.query('regression'):.0%}")  # 10%

plum.feed("revision", "regression")
plum.feed("strong_pass", "regression")
print(f"{plum.query('regression'):.0%}")  # 90%
```

## How it works

```mermaid
flowchart LR
    E["Revision or test result"] --> R["Select the requirement"]
    R --> S["Reset or update its 12-value vector"]
    S --> P["Read out pass probability"]
```

**48 state values** · `[ Regression ×12 | Build ×12 | Integration ×12 | Style ×12 ]`

The four vectors share one GRU. A revision clears its vector and resets its probability to 50%. Outdated results leave state unchanged.

The Avadhanam-inspired training streams interleave checks from all four requirements with distractors. A Bayesian simulator supplies a target probability at each step, so the model learns how to combine evidence throughout the stream.

## What it measured

![Brier error. Plum 0.11785, Bayes oracle 0.11781, latest-check rule 0.16408, last-eight-events model 0.21113. Lower is better.](assets/plum-brier.svg)

The held-out simulation used 256 streams of 192 events with 75% distractors. The Bayes oracle uses the simulator's exact reliability rates of 90% for full checks and 75% for spot checks.

**Recorded Django run.** After the agent changed the failing test, Plum held at **10%** and the eight-event model reached **50%**. The benchmark marked the run unresolved. We mapped the trace to typed events by hand. [Replay and event mapping](REAL_TRACE.md)
