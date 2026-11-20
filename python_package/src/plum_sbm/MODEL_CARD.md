# Plum model card

Plum estimates whether a requirement passes from a stream of typed evidence. It runs offline in Python with bundled weights.

- **Model** A GRU cell and readout shared across four requirement channels. Plum has 660 trainable parameters and 48 recurrent state values.
- **Input** Revisions, full check results, spot check results, and irrelevant events. The caller assigns event types and channels before inference.
- **Output** A probability for each requirement, based on the current state.
- **State updates** A revision resets its channel to 50%. Evidence on other channels leaves that state untouched. The routing rules enforce those guarantees. The recurrent cell learns how to combine applicable checks.
- **Training** Plum learned from synthetic streams with Avadhanam-inspired interleaving. Its targets came from a Bayesian simulator with 90% reliable full checks, 75% reliable spot checks, and independent evidence.
- **Evaluation** On 256 held-out synthetic streams of 192 events with 75% distractors, Plum scored 0.11785 Brier error. The eight-event model scored 0.21113, the latest-check rule 0.16408, and the exact simulator oracle 0.11781. Lower is better.
- **Limits** The model relies on structured input, fixed reliability tiers, and correct revision attribution. A recorded Django replay and a paired Click regression illustrate how it handles mapped test evidence. They give no estimate of accuracy or calibration on arbitrary agent logs.
