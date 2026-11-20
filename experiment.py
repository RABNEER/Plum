"""Small reproducible experiment in online belief tracking (Python + PyTorch)."""
import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

CHANNELS = 4
EVENTS = CHANNELS * 2 + 1  # channel/value updates, plus one distractor
UNKNOWN = 2


def generate(batch, length, distractor, seed):
    """Generate independent streams and exact state AFTER each event."""
    rng = random.Random(seed)
    x = torch.empty(batch, length, dtype=torch.long)
    y = torch.empty(batch, length, CHANNELS, dtype=torch.long)
    for b in range(batch):
        state = [UNKNOWN] * CHANNELS
        for t in range(length):
            if rng.random() < distractor:
                event = EVENTS - 1
            else:
                channel, value = rng.randrange(CHANNELS), rng.randrange(2)
                event = channel * 2 + value
                state[channel] = value
            x[b, t] = event
            y[b, t] = torch.tensor(state)
    return x, y


def window_features(events, window):
    onehot = F.one_hot(events, EVENTS).float()
    # Zero padding distinguishes missing history from every event type.
    padded = F.pad(onehot.transpose(1, 2), (window - 1, 0))
    return padded.unfold(2, window, 1).permute(0, 2, 3, 1).flatten(2)


class Stateful(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.rnn = nn.GRU(EVENTS, hidden, batch_first=True)
        self.head = nn.Linear(hidden, CHANNELS * 3)

    def forward(self, events, state=None):
        hidden, state = self.rnn(F.one_hot(events, EVENTS).float(), state)
        return self.head(hidden).reshape(*events.shape, CHANNELS, 3), state


class WindowBaseline(nn.Module):
    def __init__(self, window, hidden):
        super().__init__()
        self.window = window
        self.net = nn.Sequential(nn.Linear(EVENTS * window, hidden), nn.ReLU(),
                                 nn.Linear(hidden, CHANNELS * 3))

    def forward(self, events):
        return self.net(window_features(events, self.window)).reshape(*events.shape, CHANNELS, 3)


def scores(logits, target, events):
    prob = logits.softmax(-1)
    pred = prob.argmax(-1)
    correct = pred.eq(target)
    onehot = F.one_hot(target, 3)
    confidence = prob.max(-1).values
    # Multiclass Brier score: sum over classes, then mean over questions.
    brier = ((prob - onehot) ** 2).sum(-1).mean().item()
    ece = 0.0
    for lo in torch.arange(0, 1, 0.1):
        mask = (confidence >= lo) & (confidence < lo + 0.1 + (0.00001 if lo > 0.89 else 0))
        if mask.any():
            ece += mask.float().mean().item() * abs(correct[mask].float().mean().item() - confidence[mask].mean().item())
    distractor = (events == EVENTS - 1).unsqueeze(-1).expand_as(correct)
    return {"accuracy": correct.float().mean().item(), "brier": brier, "ece_10bin": ece,
            "distractor_accuracy": correct[distractor].float().mean().item() if distractor.any() else None,
            "last_quarter_accuracy": correct[:, events.shape[1] * 3 // 4:].float().mean().item()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--length', type=int, default=32)
    p.add_argument('--window', type=int, default=8)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='results.json')
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    models = {'stateful_gru': Stateful(args.hidden), 'fixed_window': WindowBaseline(args.window, args.hidden)}
    opts = {name: torch.optim.Adam(model.parameters(), lr=0.003) for name, model in models.items()}
    for step in range(args.steps):
        x, y = generate(args.batch, args.length, 0.35, args.seed + step)
        for name, model in models.items():
            opts[name].zero_grad()
            logits = model(x)[0] if name == 'stateful_gru' else model(x)
            loss = F.cross_entropy(logits.flatten(0, 2), y.flatten())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opts[name].step()
        if (step + 1) % 100 == 0:
            print(f'training step {step + 1}/{args.steps}', flush=True)
    results = {'config': vars(args), 'parameter_count': {n: sum(p.numel() for p in m.parameters()) for n, m in models.items()}, 'scenarios': {}}
    with torch.no_grad():
        for length, noise in [(args.length, .35), (args.length * 4, .35), (args.length * 4, .75)]:
            x, y = generate(128, length, noise, args.seed + 100000 + length + int(noise * 100))
            key = f'length_{length}_distractor_{noise}'
            results['scenarios'][key] = {}
            for name, model in models.items():
                model.eval()
                logits = model(x)[0] if name == 'stateful_gru' else model(x)
                results['scenarios'][key][name] = scores(logits, y, x)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding='utf-8')
    torch.save({'config': vars(args), 'stateful_gru': models['stateful_gru'].state_dict(),
                'fixed_window': models['fixed_window'].state_dict()}, 'checkpoint.pt')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
