import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

HERE = Path(__file__).parent
ROUNDS = HERE / "data" / "rounds"
RESULTS = HERE / "results"
MOVES = ["rock", "paper", "scissors"]

W = 8  # frames
EVAL_AT = np.round(np.arange(-0.4, 0.71, 0.1), 2)  # score
TRAIN_AT = np.arange(-0.4, 0.8, 0.025)  # train


def load_rounds() -> list[dict]:
    rounds = []
    for f in sorted(ROUNDS.glob("*.npz")):
        d = np.load(f)
        rounds.append({"t": d["t"], "kp": d["kp"], "y": MOVES.index(str(d["label"])),
                       "done_t": float(d["done_t"])})
    return rounds


def normalise(kp: np.ndarray) -> np.ndarray:
    xy = kp[:, :2] - kp[0, :2]
    return (xy / (np.linalg.norm(xy[9]) + 1e-6)).ravel()


def window(rnd: dict, cutoff: float) -> np.ndarray:
    seen = [normalise(k) for k, t in zip(rnd["kp"], rnd["t"]) if t <= cutoff and not np.isnan(k).any()][-W:]
    x = np.zeros((W, 42), dtype=np.float32)
    if seen:
        x[W - len(seen):] = np.stack(seen)
    return x.ravel()


def make_set(rounds: list[dict], cutoffs) -> tuple[mx.array, mx.array]:
    xs = [window(r, c) for r in rounds for c in cutoffs]
    ys = [r["y"] for r in rounds for _ in cutoffs]
    return mx.array(np.stack(xs)), mx.array(np.array(ys))


class Predictor(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(W * 42, hidden), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def __call__(self, x):
        return self.net(x)


def train(rounds: list[dict], steps: int, val: list[dict] | None = None, seed: int = 0):
    mx.random.seed(seed)
    x, y = make_set(rounds, TRAIN_AT)
    vx, vy = make_set(val, TRAIN_AT) if val else (None, None)

    # balance
    counts = np.bincount(np.array(y), minlength=3).astype(np.float32)
    class_w = mx.array(counts.sum() / (3 * np.maximum(counts, 1)))

    model = Predictor()
    opt = optim.AdamW(learning_rate=3e-3, weight_decay=0.01)

    def loss_fn(m, xb, yb):
        return (nn.losses.cross_entropy(m(xb), yb) * class_w[yb]).mean()

    step_fn = nn.value_and_grad(model, loss_fn)
    history = []
    for step in range(steps + 1):
        model.train()
        loss, grads = step_fn(model, x, y)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        if step % 10 == 0:
            model.eval()
            vl = loss_fn(model, vx, vy).item() if val else None
            history.append((step, loss.item(), vl))
    model.eval()
    return model, history


def accuracy(model, rounds: list[dict], cutoff: float) -> float:
    x, y = make_set(rounds, [cutoff])
    return (model(x).argmax(axis=1) == y).mean().item()


def main():
    ap = argparse.ArgumentParser(description="train the lilrps predictor and see how early it can read your throw")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    rounds = load_rounds()
    counts = np.bincount([r["y"] for r in rounds], minlength=3)
    print(f"{len(rounds)} rounds: " + ", ".join(f"{m} {c}" for m, c in zip(MOVES, counts)))
    RESULTS.mkdir(exist_ok=True)

    # crossval
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(len(rounds)), args.folds)
    acc = np.zeros((args.folds, len(EVAL_AT)))
    first_history = None
    for k, held in enumerate(folds):
        test = [rounds[i] for i in held]
        rest = [rounds[i] for i in range(len(rounds)) if i not in set(held)]
        model, history = train(rest, args.steps, val=test, seed=k)
        first_history = first_history or history
        acc[k] = [accuracy(model, test, c) for c in EVAL_AT]
        print(f"fold {k + 1}/{args.folds}: {acc[k][-1]:.0%} after landing, {acc[k][list(EVAL_AT).index(0.0)]:.0%} at shoot")

    mean = acc.mean(axis=0)
    landed = np.nanmedian([r["done_t"] for r in rounds])
    print("\nseconds after shoot -> accuracy")
    for c, a in zip(EVAL_AT, mean):
        print(f"  {c:+.1f}s  {a:.0%}")
    print(f"guessing the most common move gets {counts.max() / counts.sum():.0%}, your hand lands at {landed:+.2f}s")

    (RESULTS / "accuracy.json").write_text(json.dumps({
        "cutoffs": EVAL_AT.tolist(), "accuracy": mean.tolist(), "per_fold": acc.tolist(),
        "rounds": len(rounds), "counts": dict(zip(MOVES, counts.tolist())), "median_landing": landed,
    }, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(EVAL_AT, mean * 100, marker="o", color="#c2410c")
    ax.axhline(100 / 3, ls="--", color="grey", lw=1, label="random guess (33%)")
    ax.axvline(0, color="black", lw=1)
    ax.axvline(landed, ls=":", color="#1d4ed8", label=f"median hand lands ({landed:+.2f}s)")
    ax.set(xlabel="seconds after \"shoot\" the model stops looking", ylabel="accuracy (%)", ylim=(0, 100),
           title=f"how early lilrps reads your throw ({len(rounds)} rounds)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "accuracy.png", dpi=150)

    steps, tr, vl = zip(*first_history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, tr, label="train", color="#c2410c")
    ax.plot(steps, vl, label="held-out rounds", color="#1d4ed8")
    ax.set(xlabel="step", ylabel="loss", title="training loss (fold 1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "loss.png", dpi=150)

    # final
    model, _ = train(rounds, args.steps)
    model.save_weights(str(RESULTS / "predictor.safetensors"))
    print(f"\nsaved charts + model to {RESULTS.relative_to(HERE)}/")


if __name__ == "__main__":
    main()
