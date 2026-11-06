# lilrps

a rock paper scissors robot that reads ur hand before u even finish throwing. (building it for fallen, hack club)

**play it in ur browser: [navthings.github.io/lilrps/web](https://navthings.github.io/lilrps/web/)** (same models, runs on ur webcam, nothing gets uploaded)

![lilrps beating me 5 rounds in a row](assets/demo.gif)

## what it does

mac webcam finds my hand, a resnet18 reads the crop as rock, paper or scissors, and the moment my hand lands the robot throws whatever beats it. so it basically always wins

## run it

```
uv venv -p 3.12 .venv && uv pip install -p .venv -r requirements.txt
.venv/bin/python collect.py              # record bursts of ur hand
.venv/bin/python train.py classifier     # train on them
.venv/bin/python play.py                 # play
.venv/bin/python play.py --wheel         # play against the wheel on the pi
```

## what im doing

rn im doing a physical wheel spinny thingy, asw as cad, and ill design the circuit for the arm itself soon (pray for me)

cyall in shenzhen!!
