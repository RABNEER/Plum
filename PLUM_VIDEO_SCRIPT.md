# Plum submission film

The audience is an online judge seeing Plum for the first time. Open with the failed check and the side-by-side result. Follow with the recorded agent run, explain where Plum's state comes from, and end with a command the judge can try. Use Edge TTS Andrew Multilingual, calm pacing, and a quiet original audio bed.

| Time | Shot |
| --- | --- |
| 0:00 | Click failure and the ten later green checks |
| 0:08 | Original regression and model scores before and after the fix |
| 0:30 | Recorded Django agent run |
| 0:48 | Plum’s 48-value state |
| 1:03 | Avadhanam-inspired curriculum |
| 1:18 | Simulated recovery on a new revision |
| 1:33 | Held-out benchmark results |
| 1:50 | Public PyPI install and demo |

## hook | Open on the failure

Picture: the Click regression fails on the old commit even though the next ten checks pass. Put the actual pytest error next to that contrast.

> Ten green checks. One bug was still there. On the old version of Click, rendering help for a custom option crashed.

## compare | Reveal the paired result

Picture: show the two pinned commits and the exact same later spot check. Hold long enough to read all three columns.

> We ran Click’s own regression on the fix and its parent. Both runs ended with the same passing checks. Then we staged a completion claim and asked Plum about the original regression. Before the fix? 24.8 percent. After the fix? 96.5 percent. Each baseline gave the same score for both.

## django | Show why it matters for agents

Picture: actual browser footage from the recorded Django run. Move from the original failing check to the edited check and submission. The benchmark outcome is displayed separately.

> Here’s a recorded Django agent run. The original regression fails. The agent edits the test, sees it pass, and submits. SWE bench marked the patch unresolved. We mapped selected events by hand. To Plum, the earlier failure still counts.

## state | Explain how Plum works

Picture: three stages across the screen. Show one event entering, four requirements carrying their own state, and four probabilities coming out.

> Plum reads each event once. It carries 48 state values across four requirements. A recurrent network with 660 learned parameters weighs the checks. It runs right inside the browser.

## training | Explain the Avadhanam connection

Picture: interleave four requirement lanes. Show a distraction, a delayed query, and new evidence on a different lane.

> Avadhanam inspired the training. We interleaved evidence, added distractions, and asked about each requirement along the way. The simulator supplied the correct probabilities. The model learned which checks to combine.

## recovery | Show recovery on a new revision

Picture: actual browser footage of the separately labeled simulation. Guide the eye to the Regression tab and Plum’s live score as the revision resets, a fresh check passes, and a follow-up confirms it.

> In this separate simulation, a new revision clears old evidence. Plum returns to 50 percent. A fresh full check passes, and it rises to 90. A follow-up spot check brings it to 96.

## results | State the measured result and its scope

Picture: the four measured Brier scores, with Plum visually highlighted. Keep the exact Bayes oracle in view.

> In our held-out simulation, Plum’s Brier error was point one one eight. The eight-event baseline scored point two one one. We tested 256 streams, with three out of four events being distractors. The checks had known reliability.

## try | End with something judges can run

Picture: show the public PyPI install, the Plum demo command, and the output the judge will see.

> Try it yourself. Run pip install plum S B M, then plum demo. The package includes the weights and runs offline. Open the web demo to replay the Django run, or feed Plum your own events.
