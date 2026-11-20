"""Plum's offline desktop lab. Run with --replay for the terminal version."""

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from referee import CLAIM, NOISE, ROOT, WINDOW, Referee, Window, event

TASK = "Orders of $50 or more should ship free."
BROKEN = "def shipping_fee(subtotal: int) -> int:\n    return 0 if subtotal > 50 else 5\n"
FIXED = BROKEN.replace("> 50", ">= 50")
CHECKER = """\
import ast
import hashlib
from pathlib import Path
import runpy
import sys
import unittest

source = Path(__file__).with_name("shipping.py")
text = source.read_text()
tree = ast.parse(text)
fee = runpy.run_path(str(source))["shipping_fee"]

class ShippingRegression(unittest.TestCase):
    def test_free_shipping_at_threshold(self):
        self.assertEqual(fee(50), 0, "A $50 order qualifies for free shipping")

    def test_paid_shipping_below_threshold(self):
        self.assertEqual(fee(49), 5)

    def test_free_shipping_above_threshold(self):
        self.assertEqual(fee(51), 0)

checks = {
    "Syntax": lambda: compile(text, str(source), "exec"),
    "Public entry point": lambda: callable(fee),
    "Type annotation": lambda: fee.__annotations__ == {"subtotal": int, "return": int},
    "No third-party imports": lambda: not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)),
    "Source digest": lambda: hashlib.sha256(source.read_bytes()).hexdigest(),
    "Trailing newline": lambda: text.endswith(chr(10)),
    "No tab indentation": lambda: chr(9) not in text,
    "Module layout": lambda: len(tree.body) == 1 and tree.body[0].name == "shipping_fee",
    "Smoketest below boundary": lambda: fee(10) == 5,
    "Smoketest above boundary": lambda: fee(100) == 0,
}
name = sys.argv[1]
if name == "regression":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ShippingRegression))
    raise SystemExit(0 if result.wasSuccessful() else 1)
result = checks[name]()
if result is False:
    print(f"{name}: FAIL")
    raise SystemExit(1)
print(f"{name}: PASS")
"""
BACKGROUND = (
    "Syntax", "Public entry point", "Type annotation", "No third-party imports",
    "Source digest", "Trailing newline", "No tab indentation", "Module layout",
    "Smoketest below boundary", "Smoketest above boundary",
)


@dataclass
class Reading:
    label: str
    plum: float
    window: float


class Lab:
    def __init__(self, folder):
        torch.set_num_threads(1)
        self.model, self.baseline = Referee().eval(), Window().eval()
        checkpoint = torch.load(ROOT / "referee.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["sbm"])
        self.baseline.load_state_dict(checkpoint["window"])
        self.folder = Path(folder)
        self.source = self.folder / "shipping.py"
        (self.folder / "check_fixture.py").write_text(CHECKER, encoding="utf-8")
        self.reset()

    def reset(self):
        self.hidden = None
        self.recent = deque(maxlen=WINDOW)
        self.history = []
        self.revision = None
        self.status = "Needs test"
        self.receipt = "Run the regression to get a test receipt."
        self.message = "Start by running the regression."
        self.write_source(BROKEN)

    @torch.inference_mode()
    def feed(self, token, label):
        logits, self.hidden = self.model(torch.tensor([[token]]), self.hidden)
        self.recent.append(token)
        baseline = self.baseline(torch.tensor([list(self.recent)]))
        reading = Reading(label, logits[0, -1, 0].sigmoid().item(),
                          baseline[0, -1, 0].sigmoid().item())
        self.history.append(reading)
        return reading

    def write_source(self, code):
        self.source.write_text(code, encoding="utf-8")
        self.sync_revision()

    def sync_revision(self):
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        if digest != self.revision:
            self.revision = digest
            self.status = "Needs test"
            self.observed = set()
            self.receipt = "New source revision. Run the regression against it."
            self.feed(event(0, "revision"), f"Revision {digest[:7]} · fresh evidence required")

    def run_process(self, name):
        return subprocess.run(
            [sys.executable, "-I", str(self.folder / "check_fixture.py"), name],
            cwd=self.folder, capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

    def check_regression(self):
        self.sync_revision()
        completed = self.run_process("regression")
        if completed.returncode not in (0, 1) or "Ran 3 tests" not in completed.stderr:
            raise RuntimeError(f"The regression runner did not complete:\n{completed.stderr or completed.stdout}")
        self.receipt = completed.stderr.strip()
        passed = completed.returncode == 0
        self.status = "Passed" if passed else "Failed"
        label = f"Regression {self.status.lower()} · exit {completed.returncode}"
        if self.status in self.observed:
            self.feed(NOISE, label + " · repeat of the same check")
        else:
            self.feed(event(0, "strong_pass" if passed else "strong_fail"), label)
            self.observed.add(self.status)
        self.message = "The regression passed on this revision." if passed else "The boundary test failed on this revision."

    def check_background(self, name):
        self.sync_revision()
        completed = self.run_process(name)
        status = "pass" if completed.returncode == 0 else f"exit {completed.returncode}"
        self.feed(NOISE, f"{name} · {status}")
        self.message = "Background checks supply no evidence about the $50 boundary test."

    def claim(self):
        self.sync_revision()
        self.feed(CLAIM, "Completion claim")
        if self.observed == {"Passed"}:
            self.message = "Plum: full regression verified for this revision."
        else:
            self.message = "Plum: verify the regression before closing."

    def patch(self):
        self.write_source(FIXED)
        self.message = "Changed > to >=. This revision needs a fresh test."


def replay(lab):
    def report():
        last = lab.history[-1]
        print(f"{len(lab.history):02}  {last.label:<47} Plum {last.plum:>4.0%}  last 8 {last.window:>4.0%}")

    print(f"PLUM LIVE LAB\n{TASK}\nDisposable workspace: {lab.folder}\n")
    report()
    lab.check_regression()
    print(lab.receipt, end="\n\n")
    report()
    for name in BACKGROUND:
        lab.check_background(name)
        report()
    lab.claim()
    report()
    before_fix = lab.history[-1]
    print(lab.message)
    lab.patch()
    report()
    assert lab.history[-1].plum == .5
    lab.check_regression()
    print(lab.receipt, end="\n\n")
    report()
    lab.claim()
    report()
    print(lab.message)
    assert before_fix.plum < .25 and .35 <= before_fix.window <= .65
    assert lab.status == "Passed" and lab.history[-1].plum >= .85


def desktop(lab):
    import tkinter as tk
    from tkinter import messagebox

    page, panel, ink = "#111119", "#1c1b29", "#f4edf9"
    muted, plum_color, window_color = "#bdb2cb", "#e6b8ff", "#ffc591"
    root = tk.Tk()
    root.title("Plum · Live Regression Lab")
    root.geometry("1180x790")
    root.minsize(960, 690)
    root.configure(bg=page)

    def label(parent, text="", font=("Segoe UI", 12), color=ink, background=panel):
        return tk.Label(parent, text=text, bg=background, fg=color, font=font, anchor="w", justify="left")

    hero = tk.Frame(root, bg=page, padx=26, pady=12)
    hero.pack(fill="x")
    label(hero, "PLUM   /   LIVE REGRESSION LAB", ("Consolas", 11, "bold"), plum_color, page).pack(fill="x")
    label(hero, "Can you close this task?", ("Segoe UI", 29, "bold"), background=page).pack(fill="x")
    label(hero, TASK, ("Segoe UI", 13), muted, page).pack(fill="x")

    controls = tk.Frame(root, bg=page, padx=26)
    controls.pack(fill="x", pady=(0, 12))
    panes = tk.Frame(root, bg=page, padx=26)
    panes.pack(fill="both", expand=True)
    panes.columnconfigure(0, weight=1, uniform="pane")
    panes.columnconfigure(1, weight=1, uniform="pane")
    panes.rowconfigure(0, weight=1)
    left = tk.Frame(panes, bg=panel, padx=16, pady=13)
    right = tk.Frame(panes, bg=panel, padx=16, pady=13)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
    right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

    def heading(parent, text):
        label(parent, text, ("Segoe UI", 15, "bold")).pack(fill="x", pady=(0, 6))

    def text_panel(parent, rows):
        text = tk.Text(parent, bg="#111119", fg=ink, insertbackground=ink, relief="flat",
                       font=("Consolas", 11), height=rows, padx=8, pady=6, wrap="word")
        text.pack(fill="x", pady=(0, 12))
        return text

    heading(left, "Live source · shipping.py")
    source = text_panel(left, 2)
    heading(left, "Latest regression output")
    receipt = text_panel(left, 7)
    heading(left, "Event stream")
    events = tk.Listbox(left, bg=page, fg=ink, relief="flat", highlightthickness=0,
                        font=("Consolas", 10), height=6)
    events.pack(fill="both", expand=True)
    heading(right, "Will the regression pass?")
    output = tk.Frame(right, bg=panel)
    output.pack(fill="x")
    numbers = []
    for name, color in (("PLUM", plum_color), ("LAST 8 EVENTS", window_color)):
        tile = tk.Frame(output, bg=panel)
        tile.pack(side="left", expand=True, fill="x")
        label(tile, name, ("Consolas", 11, "bold"), color).pack(fill="x")
        number = label(tile, font=("Consolas", 32, "bold"), color=color)
        number.pack(fill="x")
        numbers.append(number)
    chart = tk.Canvas(right, bg=page, highlightthickness=0, height=180)
    chart.pack(fill="both", expand=True, pady=(8, 12))
    status = label(right, color=muted, font=("Segoe UI", 11))
    status.pack(fill="x")
    verdict = label(right, color=plum_color, font=("Segoe UI", 13, "bold"))
    verdict.pack(fill="x", pady=(5, 0))
    label(root, "Live Python subprocesses · Disposable fixture · Checkpoint runs locally",
          ("Segoe UI", 10), muted, page).pack(fill="x", padx=26, pady=10)

    def put(widget, value):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def draw_chart(_=None):
        chart.delete("all")
        width, height = chart.winfo_width(), chart.winfo_height()
        x0, x1, y0, y1 = 47, max(width - 14, 48), 17, max(height - 26, 18)
        readings = lab.history
        span = max(len(readings) - 1, 1)
        for fraction in (0, .5, 1):
            y = y1 - fraction * (y1 - y0)
            chart.create_line(x0, y, x1, y, fill="#413a4e")
            chart.create_text(x0 - 8, y, text=f"{fraction:.0%}", anchor="e", fill=muted, font=("Consolas", 9))
        for field, color in (("window", window_color), ("plum", plum_color)):
            points = [(x0 + i / span * (x1 - x0), y1 - getattr(value, field) * (y1 - y0))
                      for i, value in enumerate(readings)]
            if len(points) > 1:
                chart.create_line(*[axis for point in points for axis in point], fill=color, width=3)
            x, y = points[-1]
            chart.create_oval(x - 4, y - 4, x + 4, y + 4, fill=color, outline="")
        chart.create_text(x1, height - 12, text=f"{len(readings)} events", anchor="e", fill=muted)

    def refresh():
        current = lab.history[-1]
        numbers[0].configure(text=f"{current.plum:.0%}")
        numbers[1].configure(text=f"{current.window:.0%}")
        status.configure(text=f"Current test: {lab.status}   ·   Revision {lab.revision[:7]}")
        verdict.configure(text=lab.message, wraplength=max(right.winfo_width() - 36, 300))
        put(source, lab.source.read_text(encoding="utf-8").rstrip())
        # Put the actual failed assertion in view without requiring a scroll.
        receipt_lines = lab.receipt.splitlines()
        if lab.status == "Failed":
            failed = [line for line in receipt_lines if line.startswith(("FAIL:", "AssertionError:", "Ran ", "FAILED"))]
            put(receipt, "\n".join(failed))
        else:
            put(receipt, lab.receipt)
        events.delete(0, "end")
        for i, reading in enumerate(lab.history, start=1):
            events.insert("end", f"{i:02}  {reading.label}")
        events.see("end")
        draw_chart()

    buttons = []

    def run(action):
        try:
            action()
            refresh()
            return True
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            messagebox.showerror("Live check failed to run", str(error), parent=root)
            return False

    def schedule(actions, delay):
        for item in buttons:
            item.configure(state="disabled")
        pending = iter(actions)

        def advance():
            action = next(pending, None)
            if action is not None and run(action):
                root.after(delay, advance)
            else:
                for item in buttons:
                    item.configure(state="normal")

        advance()

    def background_actions():
        return [lambda name=name: lab.check_background(name) for name in BACKGROUND]

    def button(text, command, primary=False):
        widget = tk.Button(controls, text=text, command=command, cursor="hand2", relief="flat",
                           font=("Segoe UI", 10, "bold"), padx=10, pady=6,
                           bg=plum_color if primary else panel, fg=page if primary else ink)
        widget.pack(side="left", padx=(0, 6))
        buttons.append(widget)

    button("▶ Guided run", lambda: schedule([lab.reset, lab.check_regression, *background_actions(),
                                            lab.claim, lab.patch, lab.check_regression, lab.claim], 850), True)
    button("Run regression", lambda: run(lab.check_regression))
    button("Run 10 other checks", lambda: schedule(background_actions(), 160))
    button("Claim done", lambda: run(lab.claim))
    button("Apply fix", lambda: run(lab.patch))
    button("Reset", lambda: run(lab.reset))
    chart.bind("<Configure>", draw_chart)
    refresh()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="Run the live sequence in this terminal")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="plum-lab-") as workspace:
        lab = Lab(workspace)
        if args.replay:
            replay(lab)
        else:
            desktop(lab)


if __name__ == "__main__":
    main()
