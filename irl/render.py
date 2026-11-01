import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from cad import parts

POSES = {"rock": (95, 95), "paper": (0, 0), "scissors": (0, 95)}
COLOR = {"palm": "#d9d9d6", "cover": "#c8c8c4", "thumb": "#d9d9d6", "base": "#9a9a96",
         "yoke_a": "#3a6df0", "yoke_b": "#8f99a8"}
LIGHT = np.array([0.5, 0.6, 0.6]) / np.linalg.norm([0.5, 0.6, 0.6])


def tris(shape):
    v, f = shape.val().tessellate(0.15, 0.3)
    v = np.array([(p.x, p.y, p.z) for p in v])
    return v[np.array(f)]


def draw(ax, p, a, b, elev=20, azim=58):
    # one collection so depth sorts across parts
    allt, allc = [], []
    for name, shape in p.items():
        if name.startswith("yoke"):
            shape = shape.rotate((0, 0, 0), (1, 0, 0), -(a if name == "yoke_a" else b))
        t = tris(shape)
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        shade = 0.5 + 0.5 * np.clip(n @ LIGHT, 0, 1)
        allt.append(t)
        allc.append(np.clip(np.array(matplotlib.colors.to_rgb(COLOR[name])) * shade[:, None], 0, 1))
    ax.add_collection3d(Poly3DCollection(np.concatenate(allt), facecolors=np.concatenate(allc), linewidths=0))
    ax.set_xlim(-55, 55); ax.set_ylim(-45, 75); ax.set_zlim(-70, 75)
    ax.set_box_aspect((110, 120, 145))
    ax.view_init(elev=elev, azim=azim)
    ax.axis("off")


if __name__ == "__main__":
    p = parts()
    fig = plt.figure(figsize=(13, 5.6))
    for i, (pose, (a, b)) in enumerate(POSES.items()):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        draw(ax, p, a, b)
        ax.set_title(pose, fontsize=16, fontweight="bold", y=0.92)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "poses.png", dpi=120)
