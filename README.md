# lilrps

a rock paper scissors robot that reads your hand before you finish throwing.

mac webcam -> mediapipe finds 21 points on your hand -> a small model guesses your move from the first few frames -> robot throws the counter

right now the robot plays random. the first job is collecting rounds to train it on.

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

`--player name` tags rounds with whose hand it is (so we can test on hands it's never seen). `--record` saves the session as a video in `clips/`.

## how the referee works

no ml, just geometry. a finger counts as straight if its tip is further from the wrist than its middle knuckle. 0 straight = rock, index + middle = scissors, all 4 = paper. the thumb is ignored.

the round is "done" the first moment after shoot where the hand stops moving and holds the same move for 3 frames. that timestamp is the finish line the predictor has to beat.

if the referee keeps getting it wrong, tune `STILL` and `UP_RATIO` at the top of `play.py`. the debug line shows the live motion number.

## data

each round is one `.npz` in `data/rounds/`: every frame's 21 keypoints, times relative to shoot, the label, what the referee said, and when the hand landed.
