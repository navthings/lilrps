# lilrps

a rock paper scissors robot that reads your hand before you finish throwing. (building it for fallen, hack club)

**play it in your browser: [navthings.github.io/lilrps/web](https://navthings.github.io/lilrps/web/)** (same models, runs on your webcam, nothing gets uploaded. the setup below is for the full mac version that saves rounds)

![lilrps beating me 5 rounds in a row](assets/demo.gif)

*thats what the robot actually sees, 21 dots on my hand. the blue move is the robot locking in 0.3s after shoot, usually before my hand has even finished landing*

## what it does

mac webcam -> mediapipe (the google hand thingy) finds 21 points on your hand -> a tiny model guesses your move from the first few frames -> robot throws whatever beats it

trained on 82 rounds of my hand, it reads my throw about 0.2s before it lands and beat me 11 out of 11 live on rounds it had never seen. kinda embarrassing for me tbh

## setup

```
uv venv -p 3.12 .venv && uv pip install -p .venv -r requirements.txt
```

## play

```
.venv/bin/python play.py
```

run it from apple terminal or macos wont ask for camera permission. space starts a round, throw on "shoot!"

after each round it tells you what you threw. if its wrong press r / p / s to fix it, x bins the round, space saves it and goes again. d hides the debug line, q quits

flags:
- `--cutoff 0.3` how many seconds after shoot the robot has to commit. lower = harder for it
- `--random` robot ignores the model and throws random
- `--rules` referee uses the old finger rules instead of the hand classifier
- `--player name` tags rounds with whose hand it is
- `--record` saves the session as a video in clips/

every round gets saved, so playing = collecting data

## how it works out what you threw

v1 was no ml, just geometry. a finger counts as straight if its tip is further from your wrist than its middle knuckle. ring + pinky both straight = paper, both curled = rock or scissors, and index/middle decide which

then i found out scissors doesnt work head on. a finger pointing straight at the camera looks exactly as short as a curled one, so my V turned into rock. added a check for the gap between the index and middle tips (more than 0.8 palm widths = V) and scissors went from 73% to 85% on my saved rounds. the first version called 13 of my fists scissors lol so now the V only counts if those tips arent folded back toward your wrist

then i found the ACTUAL problem. mediapipe doesnt even see a hand head on. i ran a screenshot through it with tracking off and it found nothing, even at 5% confidence. the skeleton i was seeing was just leftover tracking from my fist, so every rule i wrote was reading a fake fist. you cant fix bad points with better rules

## hand classifier (reading pixels instead of points)

so now mediapipe only finds roughly where your hand is:

mediapipe points -> square box 2.2x bigger than the points -> 112x112 crop -> small cnn (mlx) -> rock / paper / scissors + how sure it is

![points, box and the crop the model gets](assets/journal/3_box_tracing.png)

```
.venv/bin/python collect.py
```

hold a move, press r / p / s and it saves every frame's crop for 4 seconds into data/crops/. move your hand around while it records: tilt it, go head on, near and far. about 8 bursts per move

```
.venv/bin/python train_crops.py
```

trains with random rotate / zoom / shift / mirror / blur / brightness, and tests on whole bursts it never saw (frames inside one burst are basically copies so splitting by frame would cheat). once results/classifier.safetensors exists play.py uses it as the referee, and it only trusts answers its 60%+ sure of

### v2: pretrained resnet18 + other peoples hands

the small cnn only ever saw my hand, held still. so now its a resnet18 thats already seen imagenet, fine tuned on [HaGRID](https://github.com/ai-forever/hagrid) (a hand gesture dataset with tens of thousands of different people) plus my crops weighted 3x. fist = rock, palm = paper, peace = scissors

theres also a 4th answer now, **none**. it learns it from boxes slid off the hand onto arms, shirts and background, which is exactly what my box does when mediapipe drops me. so instead of "rock 99%" on my arm it says "none" and the referee waits

```
.venv/bin/python hagrid.py      # needs the 3.5gb hagrid-sample-120k-384p zip unzipped in data/hagrid/
.venv/bin/python train_r18.py   # ~20 min on an m5, writes results/classifier_r18.safetensors
```

play.py uses classifier_r18 if it exists, otherwise the old small one. about 4ms a frame on an m5

| test | old cnn | resnet18 |
|---|---|---|
| hagrid people it never saw | - | 99.9% (1932) |
| my bursts it never saw | 100% | 100% (266) |

![hagrid crops (left 6) vs mine (right 2)](assets/journal/7_hagrid_crops.jpg)

on the 65 crops play.py saved from real rounds, by eye its right on everything i can tell what my hand is doing, except 3 close up head-on scissors it calls rock. turns out those saved labels were mostly wrong anyway (sideways scissors saved as paper, side-on fists saved as paper, my arm saved as scissors), because they came from the old referee

![its call on every real round crop. red letter = saved label, green = what it said](assets/journal/8_r18_real_rounds.jpg)

the referee also got a rule: everyones hand starts as a fist, so a fist in the first 0.45s after shoot doesnt count yet, and if paper or scissors shows up in 2 sure frames it beats rock

**status: still weak on close up head-on scissors.** next: record some head-on scissors bursts, and save raw frames from real rounds so i can test the whole referee on them

## the early guess

```
.venv/bin/python train.py
```

small mlx network that looks at the last 8 frames of hand points and guesses your move. scored with 5 fold cross validation, so every round gets tested by a model that never trained on it. prints accuracy at every cutoff from 0.4s before shoot to 0.7s after and saves the charts + results/predictor.safetensors

it scales hands by palm width and trains on copies of every round tilted toward the camera (`--aug`, default 2) so head on hands dont totally break it. on rounds tilted 60 degrees it went 59% -> 84%. the tilt is faked from mediapipe's depth guess though, so its not real head on data

![predictor before vs after](assets/journal/2_predictor.png)

it still runs on mediapipe points, so its worse head on than the classifier will be

## tuning

if the referee keeps getting stuff wrong tweak STILL, UP_RATIO, SPREAD and FOLDED at the top of play.py. the debug line shows motion and brightness and goes red when its too dark, cuz dim light = blurry hand = mediapipe drops it

best results by far: camera sees your hand side on, not fingers pointing straight at it

## data

- data/rounds/: one .npz per round. every frame's 21 points, times relative to shoot, the label, what the referee said, when your hand landed
- data/crops/&lt;move&gt;/: hand crops as jpgs, named player_burst_n.jpg (single digit n = saved by play.py from a real round)
- data/hagrid/: hagrid zip + crops, not in git

full story is in devlog.md and JOURNAL.md (summary: mediapipe hates me)
