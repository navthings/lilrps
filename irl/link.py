import json
import socket
from pathlib import Path

CONFIG = Path(__file__).parent / "config.json"
WHEEL = Path(__file__).parent / "wheel.json"
PORT = 5005
POSES = {"paper": ("open", "open"), "rock": ("closed", "closed"), "scissors": ("open", "closed")}


class Link:
    def __init__(self, host: str, config: Path, need_config: bool, fix: str):
        self.host = host
        try:
            # once, not every send
            self.addr = (socket.gethostbyname(host), PORT)
        except socket.gaierror:
            raise SystemExit(f"cant find {host}. is the pi on and on the same wifi?") from None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)
        self.sock.sendto(b"?", self.addr)
        try:
            self.sock.recvfrom(16)
        except TimeoutError:
            raise SystemExit(f"{host} didnt answer. on the pi run: sudo python3 main.py") from None
        self.sock.setblocking(False)
        self.cfg = json.loads(config.read_text()) if config.exists() else None
        if need_config and self.cfg is None:
            raise SystemExit(f"no {config.relative_to(config.parent.parent)} yet. run: {fix}")
        self.last = None

    def angles(self, a: float, b: float | None = None):
        msg = f"A{a:.0f}" if b is None else f"A{a:.0f} B{b:.0f}"
        self.sock.sendto(msg.encode(), self.addr)

    def close(self):
        self.sock.close()


class Hand(Link):
    def __init__(self, host: str = "lilrps.local", need_config: bool = True):
        super().__init__(host, CONFIG, need_config, ".venv/bin/python irl/calibrate.py")

    def pose(self, move: str):
        if move == self.last:
            return
        self.last = move
        if move == "half":
            # shortest trip to any pose
            a, b = (sum(self.cfg[s].values()) / 2 for s in "ab")
        else:
            ka, kb = POSES[move]
            a, b = self.cfg["a"][ka], self.cfg["b"][kb]
        self.angles(a, b)


class Wheel(Link):
    # one servo on A, one angle per face
    def __init__(self, host: str = "lilrps.local", need_config: bool = True):
        super().__init__(host, WHEEL, need_config, ".venv/bin/python irl/calibrate.py --wheel")

    def pose(self, move: str):
        if move == self.last:
            return
        self.last = move
        if move == "half":
            # shortest trip to any face
            deg = sorted(self.cfg.values())[1]
        else:
            deg = self.cfg[move]
        self.angles(deg)
