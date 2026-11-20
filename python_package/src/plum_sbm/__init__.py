"""Run the trained Plum belief model with no third-party dependencies."""

from functools import cache
from importlib.resources import files
import json
import math
from typing import Literal

Channel = Literal["regression", "build", "integration", "style"]
Evidence = Literal["strong_pass", "strong_fail", "spot_pass", "spot_fail"]
Kind = Literal["revision", "strong_pass", "strong_fail", "spot_pass", "spot_fail",
               "noise", "stale", "claim"]
CHANNELS: tuple[Channel, ...] = ("regression", "build", "integration", "style")
EVIDENCE: tuple[Evidence, ...] = ("strong_pass", "strong_fail", "spot_pass", "spot_fail")
IGNORED = ("noise", "stale", "claim")
WIDTH = 12

__all__ = ["CHANNELS", "Channel", "Kind", "Plum"]


@cache
def _weights():
    return json.loads(files(__package__).joinpath("weights.json").read_text(encoding="utf-8"))


def _sigmoid(value):
    return 1 / (1 + math.exp(-value))


class Plum:
    """An independent streaming session using the bundled trained weights."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Clear all four requirements."""
        self._hidden = {channel: [0.0] * WIDTH for channel in CHANNELS}

    def feed(self, kind: Kind, channel: Channel | None = None) -> None:
        """Consume one typed event. Verification and revision events require a channel."""
        if kind in IGNORED:
            if channel is not None:
                raise ValueError("Claims, stale reports and noise do not take a channel")
            return
        if kind != "revision" and kind not in EVIDENCE:
            raise ValueError(f"Unknown event kind: {kind!r}")
        if channel not in CHANNELS:
            raise ValueError(f"{kind!r} requires one of {', '.join(CHANNELS)}")
        if kind == "revision":
            self._hidden[channel] = [0.0] * WIDTH
            return

        weights = _weights()
        previous = self._hidden[channel]
        input_index = EVIDENCE.index(kind)
        inputs = [row[input_index] + bias for row, bias in zip(
            weights["rnn.weight_ih"], weights["rnn.bias_ih"])]
        recurrent = [sum(weight * value for weight, value in zip(row, previous)) + bias
                     for row, bias in zip(weights["rnn.weight_hh"], weights["rnn.bias_hh"])]
        # GRU gates are exported in reset, update, candidate order.
        updated = []
        for index, before in enumerate(previous):
            update_index = index + WIDTH
            candidate_index = index + 2 * WIDTH
            reset = _sigmoid(inputs[index] + recurrent[index])
            update = _sigmoid(inputs[update_index] + recurrent[update_index])
            candidate = math.tanh(inputs[candidate_index] + reset * recurrent[candidate_index])
            updated.append((1 - update) * candidate + update * before)
        self._hidden[channel] = updated

    def query(self, channel: Channel) -> float:
        """Estimate the probability that the selected requirement passes."""
        if channel not in CHANNELS:
            raise ValueError(f"Unknown requirement channel: {channel!r}")
        readout = _weights()["head.weight"][0]
        return _sigmoid(sum(weight * value for weight, value in zip(readout, self._hidden[channel])))

    def beliefs(self) -> dict[Channel, float]:
        """Return the current probabilities for all four channels."""
        return {channel: self.query(channel) for channel in CHANNELS}
