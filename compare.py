from collections import Counter
from pathlib import Path

import numpy as np

ROUNDS = Path(__file__).parent / "data" / "rounds"
MOVES = ["rock", "paper", "scissors"]
COUNTER = {"rock": "paper", "paper": "scissors", "scissors": "rock"}


def result(you: str, robot: str) -> str:
    if you == robot:
        return "draw"
    return "win" if COUNTER[you] == robot else "loss"


guess, react, locks, wrong = Counter(), Counter(), [], Counter()
for f in sorted(ROUNDS.glob("*.npz")):
    z = np.load(f)
    you = str(z["label"])
    if "guess" in z.files and str(z["guess"]):
        g = str(z["guess"])
        # what guess mode wouldve thrown
        guess[result(you, COUNTER[g])] += 1
        if g != you:
            wrong[(you, g)] += 1
    if "react" in z.files and bool(z["react"]):
        react[result(you, str(z["robot"]))] += 1
        locks.append(float(z["lock_t"]))


def line(name: str, c: Counter):
    n = sum(c.values())
    if n:
        print(f"{name}: {c['win']}/{n} wins ({c['win'] / n:.0%}), {c['draw']} draws, {c['loss']} losses")


line("guess mode (commits at +0.3s)", guess)
line("react mode", react)
if locks:
    print(f"react locks in at {np.median(locks) * 1000:+.0f}ms from shoot (median of {len(locks)})")
for (you, g), n in wrong.most_common(3):
    print(f"  threw {you}, guessed {g}: {n}x")
