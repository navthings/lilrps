import argparse
import json
import time
import zlib
from collections import Counter

import cv2
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

import hand
from hand import BURSTS, CLASSES, COUNTER, HERE, MOVES, REAL, RESULTS, ROUNDS, SAVE

HAGRID = hand.DATA / "hagrid"
HAGRID_RAW = HAGRID / "hagrid-sample-120k-384p"
HAGRID_CROPS = HAGRID / "crops"
IMAGENET = HAGRID / "resnet18_imagenet.safetensors"
WEB = HERE / "web" / "classifier_r18.onnx"
GESTURES = {"fist": "rock", "palm": "paper", "peace": "scissors", "peace_inverted": "scissors"}

EVAL_AT = np.round(np.arange(-0.4, 0.71, 0.1), 2)  # score
TRAIN_AT = np.arange(-0.4, 0.8, 0.025)  # train


# hagrid

def hagrid_box(b, w: int, h: int, pad: float) -> np.ndarray:
    x, y, bw, bh = b[0] * w, b[1] * h, b[2] * w, b[3] * h
    return np.array([x + bw / 2, y + bh / 2, max(bw, bh) * pad])


# box slid off the hand, like when tracking drops
def drifted(box: np.ndarray, w: int, h: int, rng) -> np.ndarray | None:
    for _ in range(10):
        ang = rng.uniform(np.pi / 3, 2 * np.pi / 3) if rng.random() < 0.6 else rng.uniform(0, 2 * np.pi)
        d = rng.uniform(0.7, 1.3) * box[2]
        cx, cy = box[0] + d * np.cos(ang), box[1] + d * np.sin(ang)
        if 0 < cx < w and 0 < cy < h:
            return np.array([cx, cy, box[2] * rng.uniform(0.8, 1.3)])
    return None


def cmd_hagrid(args):
    rng = np.random.default_rng(0)
    for c in CLASSES:
        (HAGRID_CROPS / c).mkdir(parents=True, exist_ok=True)

    counts = {c: 0 for c in CLASSES}
    none_left = args.none
    for g, move in GESTURES.items():
        ann = json.loads((HAGRID_RAW / "ann_train_val" / f"{g}.json").read_text())
        have = sorted(f.stem for f in (HAGRID_RAW / "hagrid_120k" / f"train_val_{g}").glob("*.jpg") if f.stem in ann)
        share = sum(m == move for m in GESTURES.values())
        ids = rng.permutation(have)[:args.per_move // share]
        none_here = none_left // (len(GESTURES) - list(GESTURES).index(g))
        for n, i in enumerate(ids):
            a = ann[i]
            img = cv2.imread(str(HAGRID_RAW / "hagrid_120k" / f"train_val_{g}" / f"{i}.jpg"))
            if img is None or g not in a["labels"]:  # broken
                continue
            h, w = img.shape[:2]
            box = hagrid_box(a["bboxes"][a["labels"].index(g)], w, h, rng.uniform(1.3, 1.9))
            user = a["user_id"][:12]
            cv2.imwrite(str(HAGRID_CROPS / move / f"{user}_{g}_{n}.jpg"), hand.crop(img, box))
            counts[move] += 1
            if n < none_here:
                off = drifted(box, w, h, rng)
                if off is not None:
                    cv2.imwrite(str(HAGRID_CROPS / "none" / f"{user}_{g}_{n}.jpg"), hand.crop(img, off))
                    counts["none"] += 1
                    none_left -= 1
        print(g, "->", move, counts)
    print("done", counts)


# classifier

def read(files) -> np.ndarray:
    if not files:
        return np.zeros((0, SAVE, SAVE, 3), np.uint8)
    return np.stack([cv2.resize(cv2.imread(str(f)), (SAVE, SAVE)) for f in files])


def load_hagrid():
    files, ys, val = [], [], []
    for y, c in enumerate(CLASSES):
        for f in sorted((HAGRID_CROPS / c).glob("*.jpg")):
            files.append(f)
            ys.append(y)
            val.append(zlib.crc32(f.stem.split("_")[0].encode()) % 10 == 0)  # by person
    return read(files), np.array(ys, int), np.array(val, bool)


def load_dir(root):
    files, ys, groups = [], [], []
    for y, m in enumerate(MOVES):
        for f in sorted((root / m).glob("*.jpg")):
            files.append(f)
            ys.append(y)
            groups.append(f"{m}/{f.stem.rsplit('_', 1)[0]}")  # burst
    return read(files), np.array(ys, int), np.array(groups)


def split(ys, groups, rng, frac=0.2):
    # whole bursts
    val = []
    for y in range(len(MOVES)):
        g = rng.permutation(np.unique(groups[ys == y]))
        val += list(g[:max(1, round(len(g) * frac))]) if len(g) > 1 else []
    v = np.isin(groups, val)
    return np.where(~v)[0], np.where(v)[0]


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
    if not len(imgs):
        return np.array([], int)
    return np.concatenate([np.array(model(hand.prep(imgs[i:i + 256])).argmax(axis=1))
                           for i in range(0, len(imgs), 256)])


def acc(model, imgs, ys) -> float:
    return float((predict(model, imgs) == ys).mean()) if len(ys) else float("nan")


def show(name: str, model, imgs, ys) -> float:
    conf = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t, p in zip(ys, predict(model, imgs)):
        conf[t, p] += 1
    n = conf.sum()
    a = np.trace(conf) / max(n, 1)
    print(f"\n{name}: {a:.1%} of {n}   (rows = truth, cols = said)")
    print("          " + "".join(f"{c:>9}" for c in CLASSES))
    for c, row in zip(CLASSES, conf):
        if row.sum():
            print(f"{c:>10}" + "".join(f"{v:>9}" for v in row))
    return float(a)


def cmd_classifier(args):
    for need, fix in ((HAGRID_CROPS, "run: .venv/bin/python train.py hagrid"), (IMAGENET, "download resnet18 imagenet weights from mlx-image")):
        if not need.exists():
            raise SystemExit(f"missing {need.relative_to(HERE)}. {fix}")
    rng = np.random.default_rng(0)

    t0 = time.time()
    h_imgs, h_ys, h_val = load_hagrid()
    b_imgs, b_ys, b_groups = load_dir(BURSTS)
    r_imgs, r_ys, _ = load_dir(REAL)
    if not len(b_ys):
        raise SystemExit("no bursts yet. run: .venv/bin/python collect.py")
    b_tr, b_va = split(b_ys, b_groups, rng)
    if args.all:
        b_tr = np.arange(len(b_ys))
    print(f"hagrid {len(h_ys)} ({h_val.sum()} held-out people), bursts {len(b_ys)} ({len(np.unique(b_groups))} bursts): "
          f"{len(b_tr)} train, {len(b_va)} held-out, {len(r_ys)} from real rounds. loaded in {time.time() - t0:.0f}s")

    imgs = np.concatenate([h_imgs[~h_val], b_imgs[b_tr]])
    ys = np.concatenate([h_ys[~h_val], b_ys[b_tr]])
    n_h = int((~h_val).sum())
    pool = np.concatenate([np.arange(n_h), np.repeat(np.arange(n_h, len(ys)), args.repeat)])

    mx.random.seed(0)
    model = hand.ResNet18(1000)
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
            loss, grads = step(model, hand.prep(augment(imgs[b], rng)), mx.array(ys[b]))
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            losses.append(loss.item())
        h_acc = acc(model, h_imgs[h_val], h_ys[h_val])
        b_acc = acc(model, b_imgs[b_va], b_ys[b_va])
        r_acc = acc(model, r_imgs, r_ys)
        print(f"epoch {epoch}  loss {np.mean(losses):.3f}  hagrid people {h_acc:.1%}  my bursts {b_acc:.1%}  "
              f"my real rounds {r_acc:.1%}  ({time.time() - t0:.0f}s)", flush=True)
        # real rounds stay a pure test
        score = np.nanmean([h_acc, b_acc])
        if score >= best:
            best, best_w = score, dict(tree_flatten(model.parameters()))
            mx.eval(best_w)

    model.load_weights(list(best_w.items()))
    res = {
        "hagrid_people": show("hagrid, people it never saw", model, h_imgs[h_val], h_ys[h_val]),
        "bursts": show("my bursts it never saw", model, b_imgs[b_va], b_ys[b_va]),
        "real_rounds": show("my crops from real rounds (never trained on)", model, r_imgs, r_ys),
    }
    RESULTS.mkdir(exist_ok=True)
    model.save_weights(str(hand.R18))
    (RESULTS / "classifier_r18.json").write_text(json.dumps({**res, "epochs": args.epochs, "train": int(len(pool))}, indent=2))
    print(f"\nsaved {hand.R18.relative_to(HERE)}. run train.py web to update the browser version")


# predictor

def load_rounds() -> list[dict]:
    rounds = []
    for f in sorted(ROUNDS.glob("*.npz")):
        d = np.load(f)
        rounds.append({"t": d["t"], "kp": d["kp"], "y": MOVES.index(str(d["label"])),
                       "done_t": float(d["done_t"])})
    return rounds


def tilt(rnd: dict, deg: float) -> dict:
    a = np.radians(deg)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]], dtype=np.float32)
    kp = rnd["kp"].copy()
    kp[..., 2] *= 16 / 9  # aspect
    wrist = kp[:, :1]
    return {**rnd, "kp": (kp - wrist) @ R.T + wrist}


def make_set(rounds: list[dict], cutoffs) -> tuple[mx.array, mx.array]:
    xs = [hand.window(r, c) for r in rounds for c in cutoffs]
    ys = [r["y"] for r in rounds for _ in cutoffs]
    return mx.array(np.stack(xs)), mx.array(np.array(ys))


def fit(rounds: list[dict], steps: int, val: list[dict] | None = None, seed: int = 0, aug: int = 2):
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    # fake head-on
    rounds = rounds + [tilt(r, rng.uniform(0, 70)) for r in rounds for _ in range(aug)]
    x, y = make_set(rounds, TRAIN_AT)
    vx, vy = make_set(val, TRAIN_AT) if val else (None, None)

    # balance
    counts = np.bincount(np.array(y), minlength=3).astype(np.float32)
    class_w = mx.array(counts.sum() / (3 * np.maximum(counts, 1)))

    model = hand.Predictor()
    opt = optim.AdamW(learning_rate=3e-3, weight_decay=0.01)

    def loss_fn(m, xb, yb):
        return (nn.losses.cross_entropy(m(xb), yb) * class_w[yb]).mean()

    step_fn = nn.value_and_grad(model, loss_fn)
    history = []
    for s in range(steps + 1):
        model.train()
        loss, grads = step_fn(model, x, y)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        if s % 10 == 0:
            model.eval()
            vl = loss_fn(model, vx, vy).item() if val else None
            history.append((s, loss.item(), vl))
    model.eval()
    return model, history


def guess_acc(model, rounds: list[dict], cutoff: float) -> float:
    x, y = make_set(rounds, [cutoff])
    return (model(x).argmax(axis=1) == y).mean().item()


def cmd_predictor(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = load_rounds()
    if len(rounds) < args.folds:
        raise SystemExit(f"only {len(rounds)} rounds. play more first: .venv/bin/python play.py")
    counts = np.bincount([r["y"] for r in rounds], minlength=3)
    print(f"{len(rounds)} rounds: " + ", ".join(f"{m} {c}" for m, c in zip(MOVES, counts)))
    RESULTS.mkdir(exist_ok=True)

    # crossval
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(len(rounds)), args.folds)
    res = np.zeros((args.folds, len(EVAL_AT)))
    head_on = np.zeros(args.folds)
    first_history = None
    for k, held in enumerate(folds):
        test = [rounds[i] for i in held]
        rest = [rounds[i] for i in range(len(rounds)) if i not in set(held)]
        model, history = fit(rest, args.steps, val=test, seed=k, aug=args.aug)
        first_history = first_history or history
        res[k] = [guess_acc(model, test, c) for c in EVAL_AT]
        head_on[k] = guess_acc(model, [tilt(r, 60) for r in test], 0.7)
        print(f"fold {k + 1}/{args.folds}: {res[k][-1]:.0%} after landing, {res[k][list(EVAL_AT).index(0.0)]:.0%} at shoot, "
              f"{head_on[k]:.0%} tilted 60deg")

    mean = res.mean(axis=0)
    landed = np.nanmedian([r["done_t"] for r in rounds])
    print("\nseconds after shoot -> accuracy")
    for c, a in zip(EVAL_AT, mean):
        print(f"  {c:+.1f}s  {a:.0%}")
    print(f"guessing the most common move gets {counts.max() / counts.sum():.0%}, my hand lands at {landed:+.2f}s")
    print(f"same rounds tilted 60deg toward the camera (fake head-on): {head_on.mean():.0%} after landing")

    (RESULTS / "accuracy.json").write_text(json.dumps({
        "cutoffs": EVAL_AT.tolist(), "accuracy": mean.tolist(), "per_fold": res.tolist(),
        "head_on_60deg": head_on.mean(), "aug": args.aug,
        "rounds": len(rounds), "counts": dict(zip(MOVES, counts.tolist())), "median_landing": landed,
    }, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(EVAL_AT, mean * 100, marker="o", color="#c2410c")
    ax.axhline(100 / 3, ls="--", color="grey", lw=1, label="random guess (33%)")
    ax.axvline(0, color="black", lw=1)
    ax.axvline(landed, ls=":", color="#1d4ed8", label=f"median hand lands ({landed:+.2f}s)")
    ax.set(xlabel="seconds after \"shoot\" the model stops looking", ylabel="accuracy (%)", ylim=(0, 100),
           title=f"how early lilrps reads my throw ({len(rounds)} rounds)")
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
    model, _ = fit(rounds, args.steps, aug=args.aug)
    model.save_weights(str(hand.PREDICTOR))
    print(f"\nsaved charts + model to {RESULTS.relative_to(HERE)}/")


# web

def cmd_web(args):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    w = {k: np.array(v) for k, v in mx.load(str(hand.R18)).items()}
    nodes, inits = [], []
    n = [0]

    def name(s):
        n[0] += 1
        return f"{s}_{n[0]}"

    def weight(arr, half=True):
        # fp16 on disk, fp32 to run
        k = name("w")
        if not half:
            inits.append(numpy_helper.from_array(arr.astype(np.float32), k))
            return k
        inits.append(numpy_helper.from_array(arr.astype(np.float16), k + "_h"))
        nodes.append(helper.make_node("Cast", [k + "_h"], [k], to=TensorProto.FLOAT))
        return k

    def conv_bn(x, conv, bn, stride, pad, relu=True):
        k = w[f"{conv}.weight"].transpose(0, 3, 1, 2)
        s = w[f"{bn}.weight"] / np.sqrt(w[f"{bn}.running_var"] + 1e-5)
        b = w[f"{bn}.bias"] - w[f"{bn}.running_mean"] * s
        y = name("conv")
        nodes.append(helper.make_node("Conv", [x, weight(k * s[:, None, None, None]), weight(b, False)], [y],
                                      kernel_shape=list(k.shape[2:]), strides=[stride] * 2, pads=[pad] * 4))
        if relu:
            nodes.append(helper.make_node("Relu", [y], [y + "_r"]))
            return y + "_r"
        return y

    x = conv_bn("x", "conv1", "bn1", 2, 3)
    nodes.append(helper.make_node("MaxPool", [x], ["pool"], kernel_shape=[3, 3], strides=[2, 2], pads=[1] * 4))
    x = "pool"
    for i in range(1, 5):
        for j in range(2):
            p = f"layer{i}.layers.{j}"
            stride = 2 if i > 1 and j == 0 else 1
            y = conv_bn(x, f"{p}.conv1", f"{p}.bn1", stride, 1)
            y = conv_bn(y, f"{p}.conv2", f"{p}.bn2", 1, 1, relu=False)
            skip = conv_bn(x, f"{p}.downsample.layers.0", f"{p}.downsample.layers.1", stride, 0, relu=False) \
                if f"{p}.downsample.layers.0.weight" in w else x
            a = name("add")
            nodes.append(helper.make_node("Add", [y, skip], [a]))
            nodes.append(helper.make_node("Relu", [a], [a + "_r"]))
            x = a + "_r"
    nodes.append(helper.make_node("GlobalAveragePool", [x], ["gap"]))
    nodes.append(helper.make_node("Flatten", ["gap"], ["feat"]))
    nodes.append(helper.make_node("Gemm", ["feat", weight(w["fc.weight"]), weight(w["fc.bias"], False)], ["logits"], transB=1))
    nodes.append(helper.make_node("Softmax", ["logits"], ["probs"], axis=1))

    graph = helper.make_graph(nodes, "lilrps", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 3, SAVE, SAVE])],
                              [helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["n", len(CLASSES)])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(WEB))
    print(f"saved {WEB.relative_to(HERE)} ({WEB.stat().st_size / 1e6:.1f}mb)")


# stats

def result(you: str, robot: str) -> str:
    if you == robot:
        return "draw"
    return "win" if COUNTER[you] == robot else "loss"


def cmd_stats(args):
    guess, react, locks, wrong = Counter(), Counter(), {m: [] for m in MOVES}, Counter()
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
            locks[you].append(float(z["lock_t"]))

    for name, c in (("guess mode (commits at the cutoff)", guess), ("react mode", react)):
        n = sum(c.values())
        if n:
            print(f"{name}: {c['win']}/{n} wins ({c['win'] / n:.0%}), {c['draw']} draws, {c['loss']} losses")
    for m, ls in locks.items():
        if ls:
            print(f"  {m} locks in at {np.median(ls) * 1000:+.0f}ms from shoot (median of {len(ls)})")
    for (you, g), n in wrong.most_common(3):
        print(f"  threw {you}, guessed {g}: {n}x")


def main():
    ap = argparse.ArgumentParser(description="everything lilrps trains, exports and measures")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("hagrid", help="cut hagrid photos into lilrps style crops")
    p.add_argument("--per-move", type=int, default=4000)
    p.add_argument("--none", type=int, default=5000)

    p = sub.add_parser("classifier", help="fine-tune resnet18 on hagrid + my bursts (rock / paper / scissors / none)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--repeat", type=int, default=3, help="how many times each of my crops shows up per epoch")
    p.add_argument("--all", action="store_true", help="train on every burst (held-out numbers become train numbers)")

    p = sub.add_parser("predictor", help="train the early guess and see how soon it reads my throw")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--aug", type=int, default=2, help="extra tilted copies of each round, so it learns head-on hands")

    sub.add_parser("web", help="export the classifier to onnx for the browser version")
    sub.add_parser("stats", help="guess mode vs react mode on my saved rounds")

    args = ap.parse_args()
    {"hagrid": cmd_hagrid, "classifier": cmd_classifier, "predictor": cmd_predictor,
     "web": cmd_web, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    main()
