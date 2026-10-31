from pathlib import Path

import cv2
import mlx.core as mx
import mlx.nn as nn
import numpy as np

MOVES = ["rock", "paper", "scissors"]
CLASSES = MOVES + ["none"]
SAVE = 112   # px stored
SIZE = 96    # px model
PAD = 2.2    # box / landmarks
HOLD = 1.0   # seconds
SMOOTH = 0.5
WEIGHTS = Path(__file__).parent / "results" / "classifier.safetensors"
R18 = Path(__file__).parent / "results" / "classifier_r18.safetensors"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # imagenet
STD = np.array([0.229, 0.224, 0.225], np.float32)


# held when lost
class Boxer:
    def __init__(self):
        self.box, self.seen = None, -1e9

    def update(self, kp: np.ndarray, frame: np.ndarray, now: float):
        h = frame.shape[0]
        if not np.isnan(kp).any():
            xs, ys = kp[:, 0] * h, kp[:, 1] * h
            new = np.array([(xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2,
                            max(np.ptp(xs), np.ptp(ys), 0.12 * h) * PAD])
            self.box = new if self.box is None else SMOOTH * self.box + (1 - SMOOTH) * new
            self.seen = now
        elif now - self.seen > HOLD:
            self.box = None
        return self.box


def crop(frame: np.ndarray, box) -> np.ndarray:
    cx, cy, side = box
    half = side / 2
    src = np.float32([[cx - half, cy - half], [cx + half, cy - half], [cx - half, cy + half]])
    dst = np.float32([[0, 0], [SAVE, 0], [0, SAVE]])
    return cv2.warpAffine(frame, cv2.getAffineTransform(src, dst), (SAVE, SAVE), borderMode=cv2.BORDER_REPLICATE)


def draw_box(frame: np.ndarray, box, color=(255, 200, 80)):
    cx, cy, side = box
    half = side / 2
    cv2.rectangle(frame, (int(cx - half), int(cy - half)), (int(cx + half), int(cy + half)), color, 2)


def prep(imgs: np.ndarray) -> mx.array:
    return mx.array(imgs.astype(np.float32) / 255 - 0.5)


def center(imgs: np.ndarray) -> np.ndarray:
    o = (SAVE - SIZE) // 2
    return imgs[:, o:o + SIZE, o:o + SIZE]


class HandNet(nn.Module):
    def __init__(self):
        super().__init__()
        chans = [3, 24, 48, 96, 160]
        self.convs = [nn.Conv2d(a, b, 3, padding=1) for a, b in zip(chans, chans[1:])]
        self.norms = [nn.BatchNorm(b) for b in chans[1:]]
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(chans[-1], len(MOVES))

    def __call__(self, x):
        for conv, norm in zip(self.convs, self.norms):
            x = self.pool(nn.relu(norm(conv(x))))
        return self.head(self.drop(x.mean(axis=(1, 2))))


# pretrained, same names as mlx-image
class Block(nn.Module):
    def __init__(self, a: int, b: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(a, b, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm(b)
        self.conv2 = nn.Conv2d(b, b, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm(b)
        if stride != 1 or a != b:
            self.downsample = nn.Sequential(nn.Conv2d(a, b, 1, stride, bias=False), nn.BatchNorm(b))

    def __call__(self, x):
        y = self.bn2(self.conv2(nn.relu(self.bn1(self.conv1(x)))))
        return nn.relu(y + (self.downsample(x) if "downsample" in self else x))


class ResNet18(nn.Module):
    def __init__(self, classes: int = len(CLASSES)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        chans = [64, 64, 128, 256, 512]
        for i in range(4):
            a, b = chans[i], chans[i + 1]
            setattr(self, f"layer{i + 1}", nn.Sequential(Block(a, b, 1 if i == 0 else 2), Block(b, b, 1)))
        self.fc = nn.Linear(512, classes)

    def __call__(self, x):
        x = self.maxpool(nn.relu(self.bn1(self.conv1(x))))
        for i in range(1, 5):
            x = self[f"layer{i}"](x)
        return self.fc(x.mean(axis=(1, 2)))


def prep_r18(imgs: np.ndarray) -> mx.array:
    # bgr
    x = imgs[..., ::-1].astype(np.float32) / 255
    return mx.array((x - MEAN) / STD)


class Classifier:
    def __init__(self):
        self.r18 = R18.exists()
        self.net = ResNet18() if self.r18 else HandNet()
        self.net.load_weights(str(R18 if self.r18 else WEIGHTS))
        self.net.eval()
        self.names = CLASSES if self.r18 else MOVES

    def __call__(self, img: np.ndarray) -> tuple[str, float]:
        x = prep_r18(img[None]) if self.r18 else prep(center(img[None]))
        p = mx.softmax(self.net(x), axis=-1)[0]
        i = int(p.argmax().item())
        return self.names[i], float(p[i].item())


def load() -> Classifier | None:
    return Classifier() if R18.exists() or WEIGHTS.exists() else None
