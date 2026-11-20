"use strict";

// PyTorch exports the three GRU gates in reset, update, new order.
class BeliefSession {
  constructor(weights) {
    this.weights = weights;
    this.hidden = weights.channels.map(() => new Array(weights.report.hidden_per_channel).fill(0));
    this.recent = [];
    this.logOdds = weights.channels.map(() => 0);
  }

  static sigmoid(value) {
    return 1 / (1 + Math.exp(-value));
  }

  feed(token) {
    this.recent.push(token);
    if (this.recent.length > this.weights.report.window) this.recent.shift();
    if (token < this.weights.claim) {
      const kinds = this.weights.kinds.length;
      const channel = Math.floor(token / kinds);
      const kind = token % kinds;
      if (kind === 0) this.hidden[channel].fill(0);
      else this.hidden[channel] = this.updateChannel(this.hidden[channel], kind - 1);
      this.logOdds[channel] = kind === 0 ? 0 : this.logOdds[channel] + this.weights.logOdds[kind];
    }
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
    return {
      sbm: sbmResult,
      window: windowResult,
      oracle: this.logOdds.map(sigmoid),
    };
  }
}

if (typeof module !== "undefined") module.exports = BeliefSession;
