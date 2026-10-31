import argparse
import json
from pathlib import Path

import cv2
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from hand import MOVES, SAVE, SIZE, WEIGHTS, HandNet, center, prep

HERE = Path(__file__).parent
CROPS = HERE / "data" / "crops"


def load_crops():
    imgs, ys, groups = [], [], []
    for y, move in enumerate(MOVES):
        for f in sorted((CROPS / move).glob("*.jpg")):
            imgs.append(cv2.imread(str(f)))
            ys.append(y)
            groups.append(f"{move}/{f.stem.rsplit('_', 1)[0]}")  # burst
    return np.stack(imgs), np.array(ys), np.array(groups)


def split(ys, groups, rng, frac=0.2):
    # by burst
    val = []
    for y in range(len(MOVES)):
        g = rng.permutation(np.unique(groups[ys == y]))
        val += list(g[:max(1, round(len(g) * frac))]) if len(g) > 1 else []
    v = np.isin(groups, val)
    return np.where(~v)[0], np.where(v)[0]


def augment(imgs: np.ndarray, rng) -> np.ndarray:
    out = np.empty((len(imgs), SIZE, SIZE, 3), np.uint8)
    for i, img in enumerate(imgs):
        m = cv2.getRotationMatrix2D((SAVE / 2, SAVE / 2), rng.uniform(-25, 25), rng.uniform(0.8, 1.2))
        img = cv2.warpAffine(img, m, (SAVE, SAVE), borderMode=cv2.BORDER_REPLICATE)
        x, y = rng.integers(0, SAVE - SIZE + 1, 2)
        img = img[y:y + SIZE, x:x + SIZE]
        if rng.random() < 0.5:
            img = img[:, ::-1]
        if rng.random() < 0.3:
            k = int(rng.choice([3, 5, 7]))
            img = cv2.GaussianBlur(img, (k, k), 0)
        img = img.astype(np.float32) * rng.uniform(0.6, 1.4) + rng.uniform(-25, 25)
        out[i] = np.clip(img, 0, 255).astype(np.uint8)
    return out


def refresh(model, imgs: np.ndarray, rng, batches: int = 30, bs: int = 64):
    # batchnorm stats lag the weights, catch them up before scoring
    model.train()
    for _ in range(batches):
        model(prep(center(imgs[rng.integers(0, len(imgs), bs)])))
        mx.eval(model.parameters())


def predict(model, imgs: np.ndarray) -> np.ndarray:
    model.eval()
    return np.concatenate([np.array(model(prep(center(imgs[i:i + 256]))).argmax(axis=1))
                           for i in range(0, len(imgs), 256)])


def main():
    ap = argparse.ArgumentParser(description="train the hand crop classifier (rock / paper / scissors from pixels)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    imgs, ys, groups = load_crops()
    counts = np.bincount(ys, minlength=3)
    print(f"{len(ys)} crops from {len(np.unique(groups))} bursts: " + ", ".join(f"{m} {c}" for m, c in zip(MOVES, counts)))
    rng = np.random.default_rng(0)
    tr, va = split(ys, groups, rng)
    print(f"train {len(tr)}, held-out {len(va)} (whole bursts it never trains on)")

    mx.random.seed(0)
    model = HandNet()
    opt = optim.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    class_w = mx.array(counts.sum() / (3 * np.maximum(counts, 1)), dtype=mx.float32)

    def loss_fn(m, x, y):
        return (nn.losses.cross_entropy(m(x), y) * class_w[y]).mean()

    step = nn.value_and_grad(model, loss_fn)
    best, best_w = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(tr)
        losses = []
        for i in range(0, len(order), args.batch):
            b = order[i:i + args.batch]
            loss, grads = step(model, prep(augment(imgs[b], rng)), mx.array(ys[b]))
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            losses.append(loss.item())
        refresh(model, imgs[tr], rng)
        acc = (predict(model, imgs[va]) == ys[va]).mean() if len(va) else float("nan")
        print(f"epoch {epoch:2d}  loss {np.mean(losses):.3f}  held-out {acc:.0%}")
        if len(va) == 0 or acc >= best:
            best, best_w = acc, dict(tree_flatten(model.parameters()))
            mx.eval(best_w)

    model.load_weights(list(best_w.items()))
    pred = predict(model, imgs[va]) if len(va) else np.array([], int)
    conf = np.zeros((3, 3), int)
    for t, p in zip(ys[va], pred):
        conf[t, p] += 1
    print(f"\nbest held-out {best:.0%}\nrows = what you threw, cols = what it said")
    print("          " + "".join(f"{m:>10}" for m in MOVES))
    for m, row in zip(MOVES, conf):
        print(f"{m:>10}" + "".join(f"{c:>10}" for c in row))

    WEIGHTS.parent.mkdir(exist_ok=True)
    model.save_weights(str(WEIGHTS))
    (HERE / "results" / "classifier.json").write_text(json.dumps({
        "held_out_acc": best, "confusion": conf.tolist(), "counts": dict(zip(MOVES, counts.tolist())),
        "bursts": int(len(np.unique(groups))), "epochs": args.epochs,
    }, indent=2))
    print(f"\nsaved {WEIGHTS.relative_to(HERE)}")


if __name__ == "__main__":
    main()
