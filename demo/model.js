"use strict";

// PyTorch exports the three GRU gates in reset, update, new order.
class BeliefSession {
  constructor(weights) {
    this.weights = weights;
    this.hidden = weights.channels.map(() => new Array(weights.report.hidden_per_channel).fill(0));
    this.recent = [];
    this.counts = weights.channels.map(() => new Map());
    this.seen = weights.channels.map(() => new Map());
    this.fullPass = weights.channels.map(() => false);
    this.blocked = weights.channels.map(() => false);
  }

  static sigmoid(value) {
    return 1 / (1 + Math.exp(-value));
  }

  feed(token, {evidenceId, reliability} = {}) {
    if (!Number.isInteger(token) || token < 0 || token >= this.weights.events) {
      throw new Error("Unknown event token");
    }
    const kinds = this.weights.kinds.length;
    const channel = Math.floor(token / kinds);
    const kind = token % kinds;
    const check = token < this.weights.claim && kind > 0;
    if (!check && (evidenceId !== undefined || reliability !== undefined)) {
      throw new Error("Only check results take an evidence ID or reliability");
    }
    if (check) {
      const full = kind <= 2;
      const rate = reliability === undefined ? (full ? .90 : .75) : reliability;
      if (!Number.isFinite(rate) || rate < .5 || rate >= 1) {
        throw new Error("Reliability must be a number in [0.5, 1)");
      }
      if (evidenceId !== undefined) {
        if (typeof evidenceId !== "string" || !evidenceId.trim()) {
          throw new Error("Evidence ID must be a non-empty string");
        }
        const prior = this.seen[channel].get(evidenceId);
        if (prior) {
          if (prior.kind !== kind || prior.rate !== rate) {
            this.blocked[channel] = true;
            throw new Error("Conflicting result for an existing evidence ID");
          }
          return false;
        }
        this.seen[channel].set(evidenceId, {kind, rate});
      }
      const counts = this.counts[channel];
      counts.set(rate, (counts.get(rate) ?? 0) + (kind % 2 ? 1 : -1));
      if (kind === 2) this.blocked[channel] = true;
      if (kind === 1 && evidenceId !== undefined && rate > .5) this.fullPass[channel] = true;
    }

    this.recent.push(token);
    if (this.recent.length > this.weights.report.window) this.recent.shift();
    if (token < this.weights.claim) {
      if (kind === 0) {
        this.hidden[channel].fill(0);
        this.counts[channel].clear();
        this.seen[channel].clear();
        this.fullPass[channel] = false;
        this.blocked[channel] = false;
      } else {
        this.hidden[channel] = this.updateChannel(this.hidden[channel], kind - 1);
      }
    }
    return true;
  }

  updateChannel(before, evidence) {
    const {sbm} = this.weights;
    const width = before.length;
    const fromInput = (index) => sbm["rnn.weight_ih"][index][evidence] + sbm["rnn.bias_ih"][index];
    const fromState = (index) => sbm["rnn.weight_hh"][index].reduce(
      (sum, weight, column) => sum + weight * before[column], sbm["rnn.bias_hh"][index]);
    const sigmoid = BeliefSession.sigmoid;
    return before.map((value, index) => {
      const reset = sigmoid(fromInput(index) + fromState(index));
      const update = sigmoid(fromInput(index + width) + fromState(index + width));
      const fresh = Math.tanh(fromInput(index + 2 * width) + reset * fromState(index + 2 * width));
      return (1 - update) * fresh + update * value;
    });
  }

  query() {
    const {sbm, window: baseline, report, events} = this.weights;
    const sigmoid = BeliefSession.sigmoid;
    const readout = sbm["head.weight"][0];
    const sbmResult = this.hidden.map((channelState) => sigmoid(readout.reduce(
      (sum, weight, index) => sum + weight * channelState[index], 0)));
    const offset = report.window - this.recent.length;
    const layer = baseline["net.0.weight"].map((row, index) => Math.max(0, this.recent.reduce(
      (sum, token, position) => sum + row[(offset + position) * events + token], baseline["net.0.bias"][index])));
    const windowResult = baseline["net.2.weight"].map((row, channel) => sigmoid(row.reduce(
      (sum, weight, index) => sum + weight * layer[index], baseline["net.2.bias"][channel])));
    const evidence = this.counts.map(counts => sigmoid([...counts.keys()].sort((a, b) => a - b)
      .reduce((sum, rate) => sum + counts.get(rate) * (Math.log(rate) - Math.log1p(-rate)), 0)));
    return {
      sbm: sbmResult,
      window: windowResult,
      oracle: evidence,
      ready: evidence.map((score, channel) => this.fullPass[channel] && !this.blocked[channel] && score >= .85),
      blocked: [...this.blocked],
    };
  }
}

if (typeof module !== "undefined") module.exports = BeliefSession;
