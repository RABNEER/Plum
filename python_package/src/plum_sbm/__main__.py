"""Command-line demonstrations and JSON Lines inference for Plum."""

import argparse
from contextlib import nullcontext
from importlib.metadata import version
from importlib.resources import files
import json
from pathlib import Path
import sys

from . import Plum


def demo():
    print("Plum | Two histories with the same final spot check")
    for full_check, label in (("strong_fail", "Failed"), ("strong_pass", "Passed")):
        plum = Plum()
        plum.feed("revision", "regression")
        plum.feed(full_check, "regression", evidence_id="original-regression")
        for _ in range(9):
            plum.feed("noise")
        plum.feed("spot_pass", "regression", evidence_id="spot-check")
        plum.feed("claim")
        print(f"Earlier full check {label:6} | latest spot check passed | Plum {plum.query('regression'):.1%}")


def score(source):
    plum = Plum()
    stream = nullcontext(sys.stdin) if source == "-" else Path(source).open(encoding="utf-8")
    with stream as lines:
        for index, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if (not isinstance(item, dict) or not isinstance(item.get("kind"), str)
                        or set(item) - {"kind", "channel", "evidence_id", "reliability"}):
                    raise ValueError("Expected kind and optional channel, evidence_id and reliability")
                plum.feed(**item)
            except ValueError as error:
                raise ValueError(f"Line {index}: {error}") from error
            beliefs = plum.beliefs()
            print(json.dumps({"line": index, "beliefs": beliefs,
                              "ready": {channel: plum.ready(channel) for channel in beliefs}}), flush=True)


def main():
    parser = argparse.ArgumentParser(prog="plum", description=__doc__, epilog="Run plum with no arguments for the demo.")
    parser.add_argument("--version", action="version", version=f"plum {version('plum-sbm')}")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("demo", help="Compare two streams with different full check results")
    commands.add_parser("info", help="Show the installed model card")
    stream = commands.add_parser("score", help="Read structured events and emit probabilities")
    stream.add_argument("input", help="A JSON Lines file, or - for stdin")
    args = parser.parse_args()
    if args.command in (None, "demo"):
        demo()
    elif args.command == "info":
        print(files(__package__).joinpath("MODEL_CARD.md").read_text(encoding="utf-8"), end="")
    else:
        try:
            score(args.input)
        except (OSError, ValueError) as error:
            parser.exit(2, f"plum: {error}\n")


if __name__ == "__main__":
    main()
