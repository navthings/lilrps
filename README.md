# lilrps

a rock paper scissors robot that reads your hand before you finish throwing.

mac webcam -> mediapipe finds 21 points on your hand -> a small model guesses your move from the first few frames -> robot throws the counter

trained on 82 rounds of my hand, it reads the throw about 0.2s before it finishes landing, and beat me 11 out of 11 in a live session on rounds it had never seen.

## setup

```
uv venv -p 3.12 .venv && uv pip install -p .venv -r requirements.txt
```

## play

```
.venv/bin/python play.py
```

run it from apple terminal so macos asks for camera permission. space starts a round, throw on "shoot!".

after each round the referee says what you threw. if its wrong press r / p / s to fix it, x to bin the round, space to save it and go again. d hides the debug line, q quits.

once you've run `train.py` the robot uses the trained model: at `--cutoff` seconds after shoot (default 0.3) it guesses your move from what it's seen so far and throws the counter. `--cutoff 0.1` makes it commit earlier (harder for it), `--random` turns the model off. every round still gets saved, so playing is also collecting data.

`--player name` tags rounds with whose hand it is (so we can test on hands it's never seen). `--record` saves the session as a video in `clips/`.

## how the referee works

no ml, just geometry. a finger counts as straight if its tip is further from the wrist than its middle knuckle. ring + pinky decide most of it: both straight = paper, both curled = rock or scissors, and index/middle split those two. the thumb is ignored.

the round is "done" the first moment after shoot where the palm stops moving and the same move shows up in at least 2 of 3 frames. frames where mediapipe loses the hand are skipped. that timestamp is the finish line the predictor has to beat.

if the referee keeps getting it wrong, tune `STILL` and `UP_RATIO` at the top of `play.py`. the debug line shows the live motion number.

## data

each round is one `.npz` in `data/rounds/`: every frame's 21 keypoints, times relative to shoot, the label, what the referee said, and when the hand landed.

## train

```
.venv/bin/python train.py
```

trains a small mlx network on your saved rounds and scores it with 5-fold cross validation (every round is tested by a model that never trained on it). prints accuracy at each cutoff from 0.4s before shoot to 0.7s after, and saves `results/accuracy.png`, `results/loss.png` and the model in `results/predictor.safetensors`.
