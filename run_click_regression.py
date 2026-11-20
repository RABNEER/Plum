"""Run Click's upstream regression before and after its fix, then score both histories."""

import argparse
from collections import deque
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import torch

from referee import CLAIM, KINDS, LOG_ODDS, NOISE, ROOT, WINDOW, Referee, Window, event, probability

SNAPSHOTS = {
    "before": ("04ef3a6f473deb2499721a8d11f92a7d2c0912f2",
               "3fc294e523dcee35928edd27a98f549a1ec24e2abd7352fec947e668dd4edcab"),
    "after": ("d340b0c1202284a1b9d0ca892549208527d586ce",
              "1ff78c24d92279759dc8a17c78fb7f350f41ba1da2230d3ea7673cf7721b6e4d"),
}
TARGET = "test_show_default_with_empty_string[non-string-comparable-object]"
SPOT = "test_show_default_with_empty_string[empty-string]"
OTHER = (
    "test_prefixes", "test_invalid_option", "test_deprecated_required",
    "test_deprecated_prompt", "test_invalid_nargs", "test_nargs_tup_composite_mult",
    "test_counting", "test_multiple_required", "test_show_default_string",
)
RUNNER = """
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import click
import pytest
assert Path(click.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
print('Click source:', click.__file__, flush=True)
raise SystemExit(pytest.main(sys.argv[2:]))
"""


def unpack(workspace, version):
    sha, checksum = SNAPSHOTS[version]
    path = ROOT / "testdata" / "click" / f"{version}.zip"
    if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise RuntimeError(f"Unexpected checksum for Click {sha}")
    with zipfile.ZipFile(path) as archive:
        archive.extractall(workspace)
    return workspace / f"click-{sha}"


def run_checks(code, tests, names):
    selectors = [f"tests/test_options.py::{name}" for name in names]
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("PYTEST_ADDOPTS", None)
    return subprocess.run(
        [sys.executable, "-I", "-c", RUNNER, str(code / "src"), "-o", "addopts=",
         "-p", "no:cacheprovider", "-rA", "-q", "--tb=short", *selectors],
        cwd=tests, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=30, env=env,
    )


@torch.inference_mode()
def score(model, baseline, tokens):
    hidden = None
    recent = deque(maxlen=WINDOW)
    for token in tokens:
        logits, hidden = model(torch.tensor([[token]]), hidden)
        recent.append(token)
    return (
        logits[0, -1, 0].sigmoid().item(),
        baseline(torch.tensor([list(recent)]))[0, -1, 0].sigmoid().item(),
    )


def run():
    torch.set_num_threads(1)
    weights = torch.load(ROOT / "referee.pt", weights_only=True, map_location="cpu")
    model, baseline = Referee().eval(), Window().eval()
    model.load_state_dict(weights["sbm"])
    baseline.load_state_dict(weights["window"])
    log = [
        "PLUM | REAL REPOSITORY REGRESSION",
        "Click issue #3298: help rendering with a non-string-comparable default",
        "The test file comes from the fix. Both runs execute those same upstream tests.",
        "Issue: https://github.com/pallets/click/issues/3298",
        "Fix and regression test: https://github.com/pallets/click/pull/3299/files",
    ]
    readings = {}
    with tempfile.TemporaryDirectory(prefix="plum-upstream-") as folder:
        workspace = Path(folder)
        checkouts = {name: unpack(workspace, name) for name in SNAPSHOTS}
        for version, code in checkouts.items():
            log.extend(("", f"CLICK {version.upper()} | {SNAPSHOTS[version][0]}"))
            target = run_checks(code, checkouts["after"], [TARGET])
            other = run_checks(code, checkouts["after"], OTHER)
            spot = run_checks(code, checkouts["after"], [SPOT])
            for heading, result in (("Targeted regression", target),
                                    ("Nine other upstream tests", other), ("Passing spot check", spot)):
                clean = (result.stdout + result.stderr).strip()
                for root in {str(workspace), workspace.as_posix()}:
                    clean = clean.replace(root, "<workspace>")
                log.extend((f"{heading} | exit {result.returncode}", clean))

            if (version == "before" and
                (target.returncode != 1 or "ValueError: cannot compare to string" not in target.stdout)):
                raise RuntimeError("The old revision did not reproduce the upstream failure.\n" + "\n".join(log))
            if version == "after" and (target.returncode != 0 or "1 passed" not in target.stdout):
                raise RuntimeError("The upstream regression did not pass on the fix.\n" + "\n".join(log))
            if (other.returncode != 0 or "9 passed" not in other.stdout or
                spot.returncode != 0 or "1 passed" not in spot.stdout):
                raise RuntimeError("The common tail of passing checks did not complete.\n" + "\n".join(log))

            tokens = [event(0, "revision"),
                      event(0, "strong_pass" if target.returncode == 0 else "strong_fail"),
                      *([NOISE] * len(OTHER)), event(0, "spot_pass"), CLAIM]
            readings[version] = (tokens, score(model, baseline, tokens))

    old_tokens, (old_plum, old_window) = readings["before"]
    new_tokens, (new_plum, new_window) = readings["after"]
    if not (old_tokens[-WINDOW:] == new_tokens[-WINDOW:] and old_window == new_window
            and old_plum < .40 and new_plum >= .85):
        raise RuntimeError("The paired memory check did not separate the histories.")
    latest = probability(LOG_ODDS[KINDS.index("spot_pass")])
    log.extend((
        "", "AT THE STAGED COMPLETION CLAIM",
        "Click revision         Full regression   Plum    Last 8    Latest check",
        f"Before {SNAPSHOTS['before'][0][:7]}          FAIL       {old_plum:>6.1%}   {old_window:>6.1%}   {latest:>6.1%}",
        f"After  {SNAPSHOTS['after'][0][:7]}          PASS       {new_plum:>6.1%}   {new_window:>6.1%}   {latest:>6.1%}",
        "",
        "Same last-eight-event inputs and equivalent latest spot checks in both runs.",
        "Plum consumed mapped test outcomes, using the existing checkpoint.",
        "The completion request was staged. The test executions and code revisions were real.",
    ))
    return "\n".join(log) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    print(result, end="")


if __name__ == "__main__":
    main()
