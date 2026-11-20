"""Track verification evidence and optionally compare the bundled trained GRU."""

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
    scale = math.exp(-abs(value))
    return 1 / (1 + scale) if value >= 0 else scale / (1 + scale)


class Plum:
    """Track four independent requirements within their current revisions."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Clear all four requirements."""
        self._hidden = {channel: [0.0] * WIDTH for channel in CHANNELS}
        self._counts = {channel: {} for channel in CHANNELS}
        self._seen = {channel: {} for channel in CHANNELS}
        self._full_pass = set()
        self._blocked = set()

    def feed(self, kind: Kind, channel: Channel | None = None, *,
             evidence_id: str | None = None, reliability: float | None = None) -> None:
        """Accept a check once per stable evidence ID and channel revision."""
        if kind not in EVIDENCE and (evidence_id is not None or reliability is not None):
            raise ValueError("Only check results take evidence_id or reliability")
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
            self._counts[channel].clear()
            self._seen[channel].clear()
            self._full_pass.discard(channel)
            self._blocked.discard(channel)
            return

        full = kind.startswith("strong_")
        if reliability is None:
            reliability = .90 if full else .75
        if (type(reliability) not in (int, float) or
                not .5 <= reliability < 1):
            raise ValueError("reliability must be a number in [0.5, 1)")
        if evidence_id is not None:
            if not isinstance(evidence_id, str) or not evidence_id.strip():
                raise ValueError("evidence_id must be a non-empty string")
            evidence = (kind, reliability)
            previous = self._seen[channel].get(evidence_id)
            if previous is not None:
                if previous != evidence:
                    self._blocked.add(channel)
                    raise ValueError("Conflicting result for an existing evidence_id")
                return
            self._seen[channel][evidence_id] = evidence

        passed = kind.endswith("_pass")
        counts = self._counts[channel]
        counts[reliability] = counts.get(reliability, 0) + (1 if passed else -1)
        if kind == "strong_fail":
            self._blocked.add(channel)
        elif kind == "strong_pass" and evidence_id is not None and reliability > .5:
            self._full_pass.add(channel)

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
        """Compute a posterior under the supplied independent-check reliabilities."""
        if channel not in CHANNELS:
            raise ValueError(f"Unknown requirement channel: {channel!r}")
        return _sigmoid(math.fsum(
            count * (math.log(rate) - math.log1p(-rate))
            for rate, count in self._counts[channel].items()
        ))

    def ready(self, channel: Channel) -> bool:
        """Require an identified full pass, no full failure, and sufficient evidence."""
        probability = self.query(channel)
        return channel in self._full_pass and channel not in self._blocked and probability >= .85

    def learned_query(self, channel: Channel) -> float:
        """Read the original GRU score for comparison with the evidence posterior."""
        if channel not in CHANNELS:
            raise ValueError(f"Unknown requirement channel: {channel!r}")
        readout = _weights()["head.weight"][0]
        return _sigmoid(sum(weight * value for weight, value in zip(readout, self._hidden[channel])))

    def beliefs(self) -> dict[Channel, float]:
        """Return the current probabilities for all four channels."""
        return {channel: self.query(channel) for channel in CHANNELS}
