"""Train and export Plum and its benchmark baselines."""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
CHANNELS = ("Regression", "Build", "Integration", "Style")
KINDS = ("revision", "strong_pass", "strong_fail", "spot_pass", "spot_fail")
LOG_ODDS = (0, math.log(9), -math.log(9), math.log(3), -math.log(3))
CLAIM = len(CHANNELS) * len(KINDS)
STALE = CLAIM + 1
NOISE = CLAIM + 2
EVENTS = CLAIM + 3
WINDOW = 8
HIDDEN = 48
CHANNEL_HIDDEN = HIDDEN // len(CHANNELS)
TRAIN_LENGTH = 48


def event(channel, kind):
    return channel * len(KINDS) + KINDS.index(kind)


def probability(log_odds):
    return (1 + math.tanh(log_odds / 2)) / 2


def generate(batch, length, distraction, seed):
    """Sample latent readiness and noisy reports, then label each prefix."""
    rng = random.Random(seed)
    events, targets, outcomes, ages, latest = [], [], [], [], []
    for _ in range(batch):
        truth = [rng.randrange(2) for _ in CHANNELS]
        log_odds = [0.] * len(CHANNELS)
        last_seen = [-1] * len(CHANNELS)
        last_only = [0.5] * len(CHANNELS)
        stream, beliefs, actual, age, recent = [], [], [], [], []
        for step in range(length):
            if rng.random() < distraction:
                token = rng.choice((CLAIM, STALE, NOISE))
            else:
                channel = rng.randrange(len(CHANNELS))
                if rng.random() < .20:
                    kind = 0
                    truth[channel] = rng.randrange(2)
                    log_odds[channel] = 0.
                else:
                    strong = rng.random() < .55
                    reliable = rng.random() < (.90 if strong else .75)
                    positive = truth[channel] if reliable else 1 - truth[channel]
                    kind = (1 if strong else 3) + (1 - positive)
                    log_odds[channel] += LOG_ODDS[kind]
                token = channel * len(KINDS) + kind
                last_seen[channel] = step
                last_only[channel] = probability(LOG_ODDS[kind])
            stream.append(token)
            beliefs.append([probability(value) for value in log_odds])
            actual.append(truth.copy())
            age.append([step - seen if seen >= 0 else -1 for seen in last_seen])
            recent.append(last_only.copy())
        events.append(stream)
        targets.append(beliefs)
        outcomes.append(actual)
        ages.append(age)
        latest.append(recent)
    return tuple(torch.tensor(data) for data in (events, targets, outcomes, ages, latest))


class Referee(nn.Module):
    """One learned update shared by four independent requirement states."""

    def __init__(self):
        super().__init__()
        self.rnn = nn.GRUCell(len(KINDS) - 1, CHANNEL_HIDDEN)
        self.head = nn.Linear(CHANNEL_HIDDEN, 1, bias=False)

    def forward(self, tokens, state=None):
        batch = tokens.shape[0]
        if state is None:
            state = self.head.weight.new_zeros(1, batch, HIDDEN)
        beliefs = state.reshape(batch, len(CHANNELS), CHANNEL_HIDDEN)

        kinds = tokens % len(KINDS)
        addressed = tokens < CLAIM
        channels = (tokens // len(KINDS)).clamp_max(len(CHANNELS) - 1)
        selected = F.one_hot(channels, len(CHANNELS)).bool() & addressed.unsqueeze(-1)
        updates = F.one_hot((kinds - 1).clamp_min(0), len(KINDS) - 1).to(beliefs.dtype)
        rows = torch.arange(batch, device=tokens.device)
        history = []
        for step in range(tokens.shape[1]):
            previous = beliefs[rows, channels[:, step]]
            updated = self.rnn(updates[:, step], previous)
            revised = torch.where(kinds[:, step, None] == 0, 0., updated)
            beliefs = torch.where(selected[:, step, :, None], revised[:, None, :], beliefs)
            history.append(beliefs)

        logits = self.head(torch.stack(history, dim=1)).squeeze(-1)
        return logits, beliefs.reshape(1, batch, HIDDEN)


class Window(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(EVENTS * WINDOW, 58), nn.ReLU(),
                                 nn.Linear(58, len(CHANNELS)))

    def forward(self, tokens):
        onehot = F.one_hot(tokens, EVENTS).float().transpose(1, 2)
        windows = F.pad(onehot, (WINDOW - 1, 0)).unfold(2, WINDOW, 1)
        return self.net(windows.permute(0, 2, 3, 1).flatten(2))


def metrics(prediction, target, truth, ages, events):
    error = (prediction - target).square()
    delayed = ages >= WINDOW
    delta = torch.diff(prediction, dim=1)
    target_delta = torch.diff(target, dim=1)
    affected = (events[:, 1:, None] == torch.arange(CLAIM)).reshape(
        *events[:, 1:].shape, len(CHANNELS), len(KINDS)
    ).any(-1)
    update = affected & (target_delta.abs() >= .10)
    stay = ~affected & (events[:, 1:, None] < CLAIM)
    isolation = (events[:, 1:, None] >= CLAIM).expand_as(delta)
    return {
        "oracle_mse": error.mean().item(),
        "delayed_mse": error[delayed].mean().item(),
        "brier": (prediction - truth).square().mean().item(),
        "update_error": (delta - target_delta).abs()[update].mean().item(),
        "stay_drift": delta.abs()[stay].mean().item(),
        "isolation_drift": delta.abs()[isolation].mean().item(),
    }


def evaluate(sbm, window):
    scenarios = []
    for length, distraction, seed in [(48, .35, 527501), (192, .35, 527502),
                                      (192, .75, 527503)]:
        x, y, truth, ages, latest = generate(256, length, distraction, seed)
        predictions = {
            "sbm": sbm(x)[0].sigmoid(),
            "window": window(x).sigmoid(),
            "latest": latest,
            "oracle": y,
        }
        scenarios.append({
            "events": length,
            "distractors": distraction,
            "streams": x.shape[0],
            "delayed_queries": (ages >= WINDOW).sum().item(),
            "scores": {name: metrics(p, y, truth, ages, x) for name, p in predictions.items()},
        })
    return scenarios


def serialize(model):
    return {key: tensor.tolist() for key, tensor in model.state_dict().items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.steps < 1 or args.seed < 0 or args.seed + args.steps > 527501:
        parser.error("Use positive steps and nonnegative training seeds below 527501, reserved for evaluation.")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    sbm, window = Referee(), Window()
    optimizers = [torch.optim.Adam(model.parameters(), lr=.003) for model in (sbm, window)]
    for step in range(args.steps):
        events, target, _, _, _ = generate(64, TRAIN_LENGTH, .35, args.seed + step)
        for model, optimizer in zip((sbm, window), optimizers):
            optimizer.zero_grad()
            logits = sbm(events)[0] if model is sbm else window(events)
            F.binary_cross_entropy_with_logits(logits, target).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        if (step + 1) % 200 == 0:
            print(f"Training {step + 1}/{args.steps}", flush=True)
    with torch.inference_mode():
        report = {
            "seed": args.seed,
            "steps": args.steps,
            "training_length": TRAIN_LENGTH,
            "training_distractors": .35,
            "window": WINDOW,
            "hidden": HIDDEN,
            "hidden_per_channel": CHANNEL_HIDDEN,
            "architecture": "shared_gru_with_routed_state",
            "parameter_count": {
                name: sum(p.numel() for p in model.parameters())
                for name, model in (("sbm", sbm), ("window", window))
            },
            "scenarios": evaluate(sbm.eval(), window.eval()),
        }
    weights = {
        "events": EVENTS, "channels": CHANNELS, "kinds": KINDS, "logOdds": LOG_ODDS,
        "claim": CLAIM, "stale": STALE, "noise": NOISE, "report": report,
        "sbm": serialize(sbm), "window": serialize(window),
    }
    out = args.output_dir
    (out / "demo").mkdir(parents=True, exist_ok=True)
    (out / "referee-results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save({"sbm": sbm.state_dict(), "window": window.state_dict()}, out / "referee.pt")
    (out / "demo" / "weights.js").write_text(
        "// Generated by referee.py.\nwindow.SBM_WEIGHTS = " +
        json.dumps(weights, separators=(",", ":"), allow_nan=False) + ";\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
