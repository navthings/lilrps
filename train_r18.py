import argparse
import json
import time
import zlib
from pathlib import Path

import cv2
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from hand import CLASSES, MOVES, R18, SAVE, ResNet18, prep_r18
from train_crops import split

HERE = Path(__file__).parent
NAV = HERE / "data" / "crops"
HAGRID = HERE / "data" / "hagrid" / "crops"
IMAGENET = HERE / "data" / "hagrid" / "resnet18_imagenet.safetensors"


def read(files) -> np.ndarray:
    return np.stack([cv2.resize(cv2.imread(str(f)), (SAVE, SAVE)) for f in files])


def load_hagrid():
    files, ys, val = [], [], []
    for y, c in enumerate(CLASSES):
        for f in sorted((HAGRID / c).glob("*.jpg")):
            files.append(f)
            ys.append(y)
            val.append(zlib.crc32(f.stem.split("_")[0].encode()) % 10 == 0)  # by person
    return read(files), np.array(ys), np.array(val)


def load_nav():
    files, ys, groups, real = [], [], [], []
    for y, m in enumerate(MOVES):
        for f in sorted((NAV / m).glob("*.jpg")):
            stem, idx = f.stem.rsplit("_", 1)
            files.append(f)
            ys.append(y)
            groups.append(f"{m}/{stem}")
            real.append(len(idx) == 1)  # saved by play.py
    return read(files), np.array(ys), np.array(groups), np.array(real)


# mid-throw smear
def smear(img: np.ndarray, rng) -> np.ndarray:
    k = int(rng.choice([5, 9, 13]))
    kern = np.zeros((k, k), np.float32)
    kern[k // 2] = 1
    kern = cv2.warpAffine(kern, cv2.getRotationMatrix2D(((k - 1) / 2, (k - 1) / 2), rng.uniform(0, 180), 1), (k, k))
    return cv2.filter2D(img, -1, kern / max(kern.sum(), 1e-6))


def augment(imgs: np.ndarray, rng) -> np.ndarray:
    out = np.empty_like(imgs)
    for i, img in enumerate(imgs):
        m = cv2.getRotationMatrix2D((SAVE / 2, SAVE / 2), rng.uniform(-30, 30), rng.uniform(0.75, 1.3))
        m[:, 2] += rng.uniform(-14, 14, 2)
        img = cv2.warpAffine(img, m, (SAVE, SAVE), borderMode=cv2.BORDER_REPLICATE)
        if rng.random() < 0.5:
            img = img[:, ::-1]
        if rng.random() < 0.4:
            img = smear(img, rng)
        img = img.astype(np.float32) * rng.uniform(0.6, 1.4) * rng.uniform(0.9, 1.1, 3) + rng.uniform(-25, 25)
        out[i] = np.clip(img, 0, 255).astype(np.uint8)
    return out


def predict(model, imgs: np.ndarray) -> np.ndarray:
    model.eval()
    return np.concatenate([np.array(model(prep_r18(imgs[i:i + 256])).argmax(axis=1))
                           for i in range(0, len(imgs), 256)]) if len(imgs) else np.array([], int)


def confusion(ys, pred) -> np.ndarray:
    conf = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t, p in zip(ys, pred):
        conf[t, p] += 1
    return conf


def show(name: str, conf: np.ndarray):
    n = conf.sum()
    acc = np.trace(conf) / max(n, 1)
    print(f"\n{name}: {acc:.1%} of {n}   (rows = truth, cols = said)")
    print("          " + "".join(f"{c:>9}" for c in CLASSES))
    for c, row in zip(CLASSES, conf):
        if row.sum():
            print(f"{c:>10}" + "".join(f"{v:>9}" for v in row))
    return acc


def main():
    ap = argparse.ArgumentParser(description="fine-tune a pretrained resnet18 on hagrid + nav crops (rock / paper / scissors / none)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--nav-weight", type=int, default=3, help="how many times each of your crops repeats per epoch")
    ap.add_argument("--init", action="store_true", help="keep training the current classifier_r18 instead of starting from imagenet")
    ap.add_argument("--all", action="store_true", help="train on every one of your bursts (held-out numbers become train numbers)")
    ap.add_argument("--fresh", type=int, default=0, help="bursts with an id >= this are a new session: newest one per move is always held out")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    t0 = time.time()
    h_imgs, h_ys, h_val = load_hagrid()
    n_imgs, n_ys, n_groups, n_real = load_nav()
    posed = np.where(~n_real)[0]
    tr, va = split(n_ys[posed], n_groups[posed], rng)
    n_tr, n_va, n_rl = posed[tr], posed[va], np.where(n_real)[0]
    if args.all:
        n_tr = posed
    bid = np.array([int(g.split("_")[1]) for g in n_groups])
    n_fr = np.array([], int)
    if args.fresh:
        # newest
        keep = [n_groups[posed][bid[posed] == bid[posed][n_ys[posed] == y].max()][0]
                for y in range(len(MOVES)) if (bid[posed][n_ys[posed] == y] >= args.fresh).any()]
        n_fr = posed[np.isin(n_groups[posed], keep)]
        n_tr, n_va = np.setdiff1d(n_tr, n_fr), np.setdiff1d(n_va, n_fr)
    print(f"hagrid {len(h_ys)} ({h_val.sum()} held-out people), yours {len(n_ys)}: "
          f"{len(n_tr)} train, {len(n_va)} held-out bursts, {len(n_rl)} from real rounds. loaded in {time.time() - t0:.0f}s")

    imgs = np.concatenate([h_imgs[~h_val], n_imgs[n_tr]])
    ys = np.concatenate([h_ys[~h_val], n_ys[n_tr]])
    pool = np.concatenate([np.arange((~h_val).sum()), np.repeat(np.arange((~h_val).sum(), len(ys)), args.nav_weight)])

    mx.random.seed(0)
    if args.init:
        model = ResNet18()
        model.load_weights(str(R18))
    else:
        model = ResNet18(1000)
        model.load_weights(str(IMAGENET))
        model.fc = nn.Linear(512, len(CLASSES))
    steps = args.epochs * (len(pool) // args.batch)
    opt = optim.AdamW(learning_rate=optim.cosine_decay(args.lr, steps), weight_decay=1e-4)

    def loss_fn(m, x, y):
        return nn.losses.cross_entropy(m(x), y, label_smoothing=0.1).mean()

    step = nn.value_and_grad(model, loss_fn)
    best, best_w = -1.0, None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(pool)
        losses = []
        for i in range(0, len(order) - args.batch + 1, args.batch):
            b = order[i:i + args.batch]
            loss, grads = step(model, prep_r18(augment(imgs[b], rng)), mx.array(ys[b]))
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            losses.append(loss.item())
        h_acc = (predict(model, h_imgs[h_val]) == h_ys[h_val]).mean()
        n_acc = (predict(model, n_imgs[n_va]) == n_ys[n_va]).mean()
        r_acc = (predict(model, n_imgs[n_rl]) == n_ys[n_rl]).mean()
        f_acc = (predict(model, n_imgs[n_fr]) == n_ys[n_fr]).mean() if len(n_fr) else float("nan")
        print(f"epoch {epoch}  loss {np.mean(losses):.3f}  hagrid people {h_acc:.1%}  your bursts {n_acc:.1%}  "
              f"todays bursts {f_acc:.1%}  your real rounds {r_acc:.1%}  ({time.time() - t0:.0f}s)", flush=True)
        score = (h_acc + n_acc) / 2  # real rounds stay a pure test
        if score >= best:
            best, best_w = score, dict(tree_flatten(model.parameters()))
            mx.eval(best_w)

    model.load_weights(list(best_w.items()))
    res = {
        "hagrid_people": show("hagrid, people it never saw", confusion(h_ys[h_val], predict(model, h_imgs[h_val]))),
        "nav_bursts": show("your posed bursts it never saw", confusion(n_ys[n_va], predict(model, n_imgs[n_va]))),
        "nav_fresh_bursts": show("newest burst per move, never trained on", confusion(n_ys[n_fr], predict(model, n_imgs[n_fr]))),
        "nav_real_rounds": show("your crops from real rounds (never trained on)", confusion(n_ys[n_rl], predict(model, n_imgs[n_rl]))),
    }
    model.save_weights(str(R18))
    (HERE / "results" / "classifier_r18.json").write_text(json.dumps({**res, "epochs": args.epochs, "train": int(len(pool))}, indent=2))
    print(f"\nsaved {R18.relative_to(HERE)}")


if __name__ == "__main__":
    main()
