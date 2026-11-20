# Plum · The Belief Lab

Plum is a stateful belief model trained in PyTorch to track changing evidence in an event stream. Start with the [model and training code](referee.py) or the [Python inference package](python_package/README.md).

## Install the Python package

The [`plum-sbm` package on PyPI](https://pypi.org/project/plum-sbm/) includes the trained weights and runs offline on Python 3.10 or newer with zero runtime dependencies.

```sh
pip install plum-sbm
plum demo
plum info
```

Use `from plum_sbm import Plum` to embed it in Python, or `plum score events.jsonl` to score a stream. See the [package documentation](python_package/README.md) for examples and the event schema.

## Run the local regression lab

On Windows, double-click `Open-Plum-Lab.cmd`. On any system with Python, PyTorch, and Tk installed, run `python live_lab.py` for the desktop app or `python live_lab.py --replay` for the terminal demo.

In the desktop app, choose **Guided run**, or run each step yourself. Run the regression to see the real $50 shipping test fail, run ten checks outside that requirement, and click **Claim done**. Then click **Apply fix** and rerun the regression. The app executes Python subprocesses against a disposable fixture and feeds their outcomes into the saved Plum model. The completion claim is a button press that lets you ask Plum for a judgment.

Run `python -m unittest test_live_lab -v` for the paired check. The broken and fixed versions run the same trailing checks, including one passing spot check above the $50 boundary. At the completion claim, the two runs have identical last-eight-event inputs. Plum scores them **24.8%** and **96.5%** respectively. The eight-event model outputs **77.3%** in both cases, and the latest-check rule outputs **75.0%** in both cases. These are model scores from one controlled fixture with structured test events.

## Reproduce a real Click regression

With pytest installed, run `python run_click_regression.py --output submission/Plum-Click-Test.txt`. It runs the same test from [Click PR #3299](https://github.com/pallets/click/pull/3299/files) against the fix and its parent. The actual fix and test are upstream. Their BSD license is included in the two pinned source archives in `testdata/click/`. In both runs the targeted check is followed by nine passing upstream tests and a passing spot check. The staged completion claim asks Plum whether the original regression is resolved. See the [test output](submission/Plum-Click-Test.txt) and [comparison image](submission/Plum-Click-Evidence.png).

## Try the demo

The live demo opens with a recorded mini-SWE-agent run where the agent submitted a Django patch that SWE-bench Verified marked unresolved. Inference runs in your browser using the exported PyTorch weights. The page keeps the full timeline for display. Plum receives each token once and carries 48 recurrent values, 12 per requirement. [Open the live Plum demo](https://plum-sbm-demo.vercel.app/)

In **Recorded Django run**, step to event 3 to see the original regression test fail, event 7 to see the edited test pass, and event 17 to see the submission. Plum gives the original check a 10% chance of passing at submission. The eight-event baseline returns to 50%. We mapped selected transcript entries by hand and display the recorded benchmark outcome separately from the model output. [Source and event mapping](REAL_TRACE.md)

In **Simulated failure and recovery**, step to event 12 for the premature completion claim, then advance through a fresh revision and passing checks. In **Conflicting evidence**, a full check failure outweighs a passing spot check.

## Train and verify

PyTorch is needed for training. The saved run used Python 3.14.3 and PyTorch 2.11.0 CPU. The inference parity test uses Node.js to compare the JavaScript runtime with PyTorch.

```sh
python -m pip install torch
python referee.py
python -m unittest -v
```

`referee.py` writes the trained checkpoint to `referee.pt`, the held-out results to `referee-results.json`, and the browser weights to `demo/weights.js`. Set `--output-dir` to keep a trial run separate from the published checkpoint. The parity test compares both exported networks to their PyTorch checkpoints.

## Evaluation

The simulator models four independent requirements. An applicable full check has 90% sensitivity and specificity. A spot check has 75%. Tests are conditionally independent in the simulator. A new revision replaces the latent readiness with a fresh draw and resets that requirement’s prior to 0.5. Outdated results, agent claims, and unrelated activity are separate event types and carry no information about readiness. The oracle updates log odds using the known check rates and provides posterior targets at every event.

Plum shares a trained GRU cell and readout across requirements. Each applicable check updates only its own requirement. Revision events clear the corresponding state and return its probability to 50%. Out-of-scope events skip the state update. Those guarantees come from the routing and reset rules. The recurrent cell learns how to combine checks.

Plum and the eight-event baseline train on the same 48-event streams using binary cross-entropy against those soft targets. The evaluation compares them with a latest-check rule (retains one check per requirement and uses its reliability as the probability) and the exact Bayes oracle. The oracle gets the simulator's true update rules. Evaluation seeds are held apart from training and report 256 independently generated streams per condition.

| On held-out 192-event streams with 75% distractors | Brier ↓ | Delayed MSE ↓ | Update error ↓ | Stay drift ↓ | Isolation drift ↓ |
|---|---:|---:|---:|---:|---:|
| Plum, 660 parameters | 0.11785 | 0.000012 | 0.00238 | 0 | 0 |
| Eight-event MLP, 10,966 parameters | 0.21113 | 0.144503 | 0.12636 | 0.02520 | 0.02165 |
| Latest-check rule | 0.16408 | 0.049285 | 0.11977 | 0 | 0 |
| Exact Bayes oracle | 0.11781 | 0 | 0 | 0 | 0 |

Brier is squared error against the sampled binary outcomes. Delayed MSE measures error against the oracle on questions with evidence at least eight events old. Update error measures how closely the model matches a material oracle change (at least 10 percentage points). Stay drift measures movement on unrelated channels after an in-scope event. Isolation drift measures movement on all channels after an out-of-scope event.

The results describe one training run on synthetic data. The simulator has known reliability rates, clearly marked stale events, and independent requirement channels. The recorded run demonstrates one hand-mapped replay. It gives no estimate of accuracy or calibration on raw agent logs. The [independent verdict](VERDICT.md) evaluated an earlier checkpoint.

See the [Devpost draft](SUBMISSION.md) and [pitch video shot plan](PLUM_VIDEO_SCRIPT.md).

## Original symbolic experiment

A first controlled experiment, not a natural-language agent referee. Four channels independently hold an unknown, 0, or 1 state. Events update one channel or are irrelevant distractors. The symbolic oracle labels all channels after each event. A GRU reads each event once, retaining a fixed-size hidden state; a stateless MLP reads only the latest eight one-hot events. Both output a three-class distribution for each channel at every step.

## Run

Requires Python and PyTorch (tested with Python 3.14 and PyTorch 2.11 CPU).

```sh
python -m unittest -v
python experiment.py
```

`results.json` contains configuration, parameter counts, accuracy, multiclass Brier score, ten-bin expected calibration error (ECE), accuracy at distractor steps, and accuracy in the final quarter of each stream. Lower Brier and ECE are better. `checkpoint.pt` stores both model weights. Use `--steps`, `--seed`, `--window`, `--length`, and `--output` to change the experiment. The default trains on length 32 with 35% distractors, then evaluates independently generated length 32 and length 128 streams, including a 75% distractor shift. Training and evaluation use separate deterministic seeds.

At inference `Stateful.forward(events, state)` returns logits and the GRU state, enabling incremental feeding. The state size is constant in stream length; the training implementation batches complete streams for efficiency. Outputs are probabilities after softmax, not guaranteed calibrated. The baseline uses the same events and query labels, but necessarily discards history outside its fixed window; this measures one memory/cost tradeoff, not superiority to all alternatives. The models are not parameter- or compute-matched; parameter counts are reported. A deterministic symbolic state tracker solves this toy environment exactly and is a necessary reference for its limitations.

Events are discrete symbols, updates are noiseless, and labels are deterministic. This does not establish natural-language understanding, prompt-injection resistance, probabilistic evidence integration, genuine uncertainty, real agent-session performance, or a novel architecture. A learned GRU is an established architecture. Further work should introduce partial/noisy evidence and probabilistic oracle labels, meaningful distractors, unseen event semantics, strong memory/retrieval baselines, multiple seeds and confidence intervals, and independently labeled real traces.
