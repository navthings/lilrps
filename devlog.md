# devlog

## 29 oct 2026

picked the idea: a rock paper scissors robot that reads your hand mid-throw and counters it before you finish.

big problem was knowing when a throw is actually done and what it ended on. solved it with a referee that isn't ml at all: count straight fingers from mediapipe's keypoints, and call the throw done when the hand stops moving and holds the same shape for 3 frames.

first version of the game is in. robot plays random for now. next: collect rounds, check how often the referee is right.

first 9 rounds: referee got every rock and paper, missed both scissors. replayed the saved rounds to find out why: on scissors mediapipe keeps losing the hand and the fingertips jitter, so it never looked "still". fixed it by checking stillness on the palm only, skipping dropped frames, letting one glitchy frame through, and using the full hand model. replay went 7/9 to 8/9.

the full hand model was a mistake: fps dropped from ~20 to 15 and it lost my fist mid-countdown and never found it again, so rock broke. back to the lite model with lower detection confidence so it grabs a lost hand back faster.
