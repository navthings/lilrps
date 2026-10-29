# devlog

## 29 oct 2026

picked the idea: a rock paper scissors robot that reads your hand mid-throw and counters it before you finish.

big problem was knowing when a throw is actually done and what it ended on. solved it with a referee that isn't ml at all: count straight fingers from mediapipe's keypoints, and call the throw done when the hand stops moving and holds the same shape for 3 frames.

first version of the game is in. robot plays random for now. next: collect rounds, check how often the referee is right.
