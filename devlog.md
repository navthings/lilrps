# devlog

## 29 oct 2026

picked the idea: a rock paper scissors robot that reads your hand mid-throw and counters it before you finish.

big problem was knowing when a throw is actually done and what it ended on. solved it with a referee that isn't ml at all: count straight fingers from mediapipe's keypoints, and call the throw done when the hand stops moving and holds the same shape for 3 frames.

first version of the game is in. robot plays random for now. next: collect rounds, check how often the referee is right.

first 9 rounds: referee got every rock and paper, missed both scissors. replayed the saved rounds to find out why: on scissors mediapipe keeps losing the hand and the fingertips jitter, so it never looked "still". fixed it by checking stillness on the palm only, skipping dropped frames, letting one glitchy frame through, and using the full hand model. replay went 7/9 to 8/9.

the full hand model was a mistake: fps dropped from ~20 to 15 and it lost my fist mid-countdown and never found it again, so rock broke. back to the lite model with lower detection confidence so it grabs a lost hand back faster.

first training run on 61 rounds (only 9 rock). small mlx network looking at the last 8 frames of hand keypoints, scored with 5-fold cross validation. before "shoot" it's at chance, which makes sense since my fist is just pumping. it climbs fast after: 70% at +0.2s, 83% by the time my hand usually lands (+0.41s). so it's reading the throw about 0.2s before i finish. next: way more rock, then plug it into the game.

plugged the model into the game. the robot now has to commit at a cutoff (0.3s after shoot by default) using only the frames it's seen, then throws the counter. it shows its move on screen while my hand is still landing.

recorded it. against the random robot: lost 2, drew 1. against the trained model on fresh rounds: it won 11 out of 11, and in 8 of them it called my move before my hand had even finished landing. also found the recordings were playing back too fast (the camera does ~15fps while recording but the video said 21), so recording now writes frames by the clock.
