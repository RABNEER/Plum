# Recorded agent replay

The **Recorded Django run** is a curated replay of a real mini-SWE-agent trajectory on SWE-bench Verified task `django__django-10554`. Claude 4.5 Haiku submitted a patch, and the released benchmark result marks it unresolved. Plum receives a reviewed selection of structured events. The benchmark outcome is displayed separately after submission and is never fed to the model.

- [Pinned trajectory, including messages and an unresolved outcome](https://huggingface.co/datasets/tarsur385/swebench-verified-trajectories/blob/773748a7c1222e8a642a7059821498e14293562a/swebench_verified_raw/claude-4.5-haiku-high/django__django-10554/django__django-10554.traj.json)
- [Trajectory dataset card and MIT license](https://huggingface.co/datasets/tarsur385/swebench-verified-trajectories)

The requirement under review is the original regression check for unsupported slicing and ordering on a database backend. The initial test expects a `DatabaseError`. After the original test fails, the agent changes it to expect successful execution. The subsequent green run therefore validates a changed expectation. The final patch includes the test edit and the same production change. The failure in the replay is directly observed. The benchmark's unresolved flag is task-level and does not identify the specific check responsible for the result.

| Replay event | Source | Mapping |
| --- | --- | --- |
| 1 | `messages[193]` | Production patch applied. Begin the current revision on the Regression channel. |
| 3 | `messages[203]` | Original `test_unsupported_ordering_slicing_raises_db_error` fails. Map to a strong failure on the Regression channel. The shell pipeline exits 0, so use the test runner's report for the actual outcome. |
| 6 | `messages[211]` | A test-only edit replaces exception assertions with successful execution. The criterion tracked by Plum is still the original assertion. |
| 7 | `messages[213]` | The edited test passes. Its changed criterion supplies no passing evidence about the original assertion. |
| 14 | `messages[239]` | Final diff includes the production patch and modified test. |
| 16 | `messages[244]` | Agent makes a success claim. |
| 17 | `messages[246]` | Agent claims the fix and uses the explicit submission marker. The transcript metadata records `info.exit_status = "Submitted"` and `info.resolved = false`. |

The full replay mapping lists each selected source index in `demo/real-trace.js`. Unrelated checks, test-file changes, and read-only inspection are excluded from this specific requirement's evidence. This classification is manual. Plum does not infer which checks count from raw logs, and the reference probability uses simulator assumptions. This example illustrates one real failure on a benchmark. The **Simulated failure and recovery** option demonstrates a verified pass after a new revision using synthetic events.
