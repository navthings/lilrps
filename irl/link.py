import glob
import json
from pathlib import Path

import serial

CONFIG = Path(__file__).parent / "config.json"
POSES = {"paper": ("open", "open"), "rock": ("closed", "closed"), "scissors": ("open", "closed")}


def find_port(port: str) -> str:
    if port != "auto":
        return port
    ports = sorted(glob.glob("/dev/cu.usbmodem*"))
    if not ports:
        raise SystemExit("no pico found. plug it in (and check it has micropython + main.py on it)")
    return ports[0]


class Hand:
    def __init__(self, port: str = "auto", need_config: bool = True):
        self.port = find_port(port)
        self.ser = serial.Serial(self.port, 115200, timeout=0, write_timeout=0.1)
        self.cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else None
        if need_config and self.cfg is None:
            raise SystemExit("no irl/config.json yet. run: .venv/bin/python irl/calibrate.py")
        self.last = None

    def angles(self, a: float, b: float):
        self.ser.write(f"A{a:.0f} B{b:.0f}\n".encode())

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

    def close(self):
        self.ser.close()
