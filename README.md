# Retail Behavioural Video Analytics

An end-to-end, **fully local** video analytics pipeline that measures customer
interest, shelf engagement and staff–customer interaction from fixed retail
CCTV footage.

Submission for the Hendricks *Computer Vision Engineer* take-home assessment.

| | Source video | Metric | Deliverables |
|---|---|---|---|
| **Task 1** | `entrance.mp4` | Total Interested / Interested Entered / Interested Passed By | `entrance_annotated.mp4` + `task1_store_interest.csv` |
| **Task 2** | `interior.mp4` | Interest events per shelf A–D | `interior_annotated.mp4` + `task2_shelf_interest.csv` |
| **Task 3** | `entrance.mp4` | Average interaction sessions per staff instance | *same* `entrance_annotated.mp4` + `task3_staff_interactions.csv` |

### Results on the supplied footage

Produced by `python -m rva.cli all` on the two 7.5-minute clips, CPU-only,
with the configuration exactly as committed. Every number below also appears
as a text overlay in the corresponding annotated video.

| Task 1 — `entrance.mp4` | |
|---|---:|
| Total Interested | **6** |
| Interested Entered | **2** |
| Interested Passed By | **4** |

| Task 2 — `interior.mp4` | events |
|---|---:|
| Shelf A | **6** |
| Shelf B | **4** |
| Shelf C | **2** |
| Shelf D | **4** |
| total | 16 |

| Task 3 — `entrance.mp4` | |
|---|---:|
| Staff instances detected | 5 |
| Total interaction sessions | 5 |
| **Average sessions per staff member** | **1.00** |

Task 3 detail (staff instances with zero sessions are included in the average,
as the brief requires):

| Staff instance | in view | sessions |
|---|---|---:|
| Staff 1 | 0.0 – 18.0 s | 0 |
| Staff 2 | 66.1 – 88.9 s | 0 |
| Staff 3 | 77.0 – 88.2 s | 0 |
| Staff 4 | 247.0 – 287.8 s | 5 |
| Staff 5 | 249.2 – 261.7 s | 0 |
| | **average** | **(0+0+0+5+0)/5 = 1.00** |

One further person entered the store during the clip without any observable
prior interest (audit CSV row `ENTERED_NO_PRIOR_INTEREST`). They are correctly
absent from all three Task 1 counts — the brief asks for *interested* people who
*subsequently* entered — and they are reported in the audit trail rather than
quietly dropped.

Sensitivity of these numbers to the chosen thresholds is measured in §4.4.

---

## 1. Quick start

### Docker (recommended — this is the reproducible path)

```bash
# put the two source videos in ./data/
#   data/entrance.mp4
#   data/interior.mp4

docker build -t rva .            # downloads model weights, runs the unit tests
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/outputs:/app/outputs" \
  rva all
```

### Local Python (3.10+)

```bash
python -m pip install -r requirements.txt
bash models/download_models.sh          # pretrained YOLO11-pose weights
export PYTHONPATH=src

python -m rva.cli all                   # both videos, all three tasks
```

or with the Makefile: `make setup && make run`.

The annotated videos are written with OpenCV's `mp4v` FourCC, which is the one
codec present in every OpenCV build on every platform — but it is an old MPEG-4
encoder and the files are ~8× larger than they need to be. `bash
tools/compress_outputs.sh` re-encodes them to H.264 (415 MB → 52 MB for the
entrance clip, visually identical). It is a separate optional step so that a
reviewer without ffmpeg can still run the whole pipeline. **The videos in this
submission have already been through it.**

### Useful variants

```bash
python -m rva.cli entrance                       # Task 1 + Task 3 only
python -m rva.cli interior                       # Task 2 only
python -m rva.cli entrance --max-seconds 60      # 60-second smoke test
python -m rva.cli zones --config configs/entrance.yaml \
       --video data/entrance.mp4 --out outputs/zones.jpg   # verify the scene setup
python -m rva.cli grid --video data/interior.mp4 --at 25   # coordinate grid for re-drawing zones

# any config value can be overridden without editing YAML
python -m rva.cli entrance --set task1.interest.score_threshold=1.4
python -m rva.cli all --set model.device=cuda --set model.weights=models/yolo11m-pose.pt
```

### Re-tuning any threshold in seconds (the feature cache)

Inference is the expensive part of the pipeline, and *none* of it depends on
the thresholds a reviewer might want to question. So a run can cache the
per-frame detections, keypoints and appearance measurements, and a `replay`
then feeds that cache through the same task state machines with a different
config — recomputing every count in about two seconds instead of an hour:

```bash
python -m rva.cli entrance --dump-features outputs/cache/entrance.jsonl

python -m rva.cli replay --features outputs/cache/entrance.jsonl \
       --config configs/entrance.yaml \
       --set task1.interest.score_threshold=1.40
```

Replaying an unchanged config reproduces the original numbers exactly — a
property covered by `tests/test_replay.py`. This is what makes the threshold
choices *auditable* rather than merely documented: any claim of the form
"raising X to Y would change the answer to Z" can be checked directly.

### Runtime

Defaults are tuned for **CPU-only** execution. On 2 CPU cores the full run is
roughly 45 minutes per video at `frame_stride: 3`. On a CUDA GPU
(`--set model.device=cuda`) it is a few minutes. To trade accuracy for speed:

```bash
python -m rva.cli all --set model.weights=models/yolo11n-pose.pt \
                      --set model.imgsz=768 --set runtime.frame_stride=5
```

Temporal rules are expressed in **seconds**, never frames, so changing
`frame_stride` cannot change a count.

---

## 2. Technologies

| Component | Choice | Why |
|---|---|---|
| Detection + pose | **Ultralytics YOLO11-pose** (`yolo11s-pose.pt`, COCO-pretrained, no fine-tuning) | One forward pass gives the box **and** 17 keypoints. Keypoints are needed three times over — body orientation, reach direction, and the torso crop for the staff classifier — so a pose model is strictly cheaper than a detector plus a separate pose network, and box↔keypoint association is free. |
| Tracking | **BoT-SORT**, tuned in `configs/botsort_rva.yaml` | Kalman motion model + appearance. Three deliberate changes from the defaults: `gmc_method: none` (both cameras are bolted to a ceiling, so global motion compensation has no ego-motion to remove and is instead driven by the *people* moving through frame, injecting error); `track_buffer: 60` (~6 s at our 10 Hz feed, enough to survive a shopper walking behind a gondola); `new_track_thresh: 0.35` (spawning a new identity is what corrupts a unique-person count, so a marginal detection should die rather than become a new person). |
| Everything else | NumPy + OpenCV | Zone geometry, orientation fusion, the event state machines, the staff classifier and the annotation layer are all written here, so every decision is inspectable and explainable. |

`imgsz: 960` rather than the usual 640: passers-by in the mall walkway are only
60–110 px tall and recall on them collapses at 640.

**No external or paid inference services are used.** All weights are public
Ultralytics COCO checkpoints, downloaded at setup time and executed locally.

---

## 3. Scene understanding

The methodology only makes sense against what is actually in the footage. Both
clips are 1280×720, 30 fps, ≈7.5 min, near-simultaneous on 2023-11-07.

### `entrance.mp4` — the camera is *inside* the store looking out

Three horizontal bands:

| rows | content |
|---|---|
| y ≈ 0–215 | **atrium void** — glass balustrade, escalator, and the mall level *below* seen through the glass. Different floor: must never be counted. |
| y ≈ 210–455 | **mall walkway** — grey tile with dark diamond inserts. The passers-by Task 1 is about. |
| y ≈ 290–720 | **store interior** — cream tile, navy carpet lounge, display plinths. |

- There is **no door**: the frontage is open, so the "entrance boundary" is a
  material transition in the floor (the lease line), drawn by hand as
  `scene.storefront_line`.
- Staff wear a **dark navy bib apron with a small circular light badge**, and
  work close to the camera (120–250 px tall), so the badge is resolvable —
  but the armchairs and carpet are *the same navy*, so appearance alone is not
  enough.
- The hard case is at the **bottom frame edge**, where shoe fittings happen:
  the staff member kneels, is truncated by the border and occluded by the
  armchair, and pose keypoints are unreliable there.

### `interior.mp4` — high corner camera, strong barrel distortion

Figure 1's shelf labels map onto the fixtures as: **A** the right-hand gondola
with the sock rack, **B** the low front-centre gondola with the orange shoebox
rack, **C** the long tiered display across the middle, **D** the shoe wall down
the left. The brief supplies label anchors only; the polygons are ours.

- Lens distortion and the absence of any known floor reference rule out a
  ground-plane homography (see §4.1).
- Customers routinely stand in the aisle **between** B and A, and between B and
  D — the ambiguity the brief calls out.
- The far band (y < 230) is the service counter on the other side of the store;
  those people are not engaging with A–D and are excluded by the ROI.

### How the zones were calibrated

Not by eye. The detector was run over **both complete clips** at 2 Hz and every
foot position plotted onto a reference frame (3,017 detections for entrance,
1,357 for interior). Two things fell out:

1. Essentially **no** detections above y ≈ 225 in the entrance view, confirming
   the ROI cut is safe.
2. A clear **dead band** between walkway traffic and in-store traffic, caused by
   the store's front-line fixtures. The storefront line is drawn *through that
   dead band*, which makes its exact placement far less sensitive than it looks.

Re-verify at any time with `python -m rva.cli zones ...`, which renders the
configured polygons onto a real frame.

---

## 3b. Pipeline architecture — and why it is two passes

The obvious design is a single loop: detect, decide, count, draw. It is wrong
for this footage, and the reason is worth stating because it is a correctness
issue rather than a stylistic one.

Whether a track is *staff* can only be judged from the whole clip. The badge
detection rate and the apron average are least settled precisely in a track's
first few seconds — which is exactly when a single-pass design has to commit. A
first implementation did commit, and produced a role that changed mid-clip: the
on-screen counters, the CSVs and the audit trail then disagreed about who was
who. The brief requires the CSV values to match the annotated video, so this had
to be fixed structurally rather than patched.

```
        ┌──────────────────────────────────────────────────────────────┐
 pass 1 │  decode → YOLO11-pose → BoT-SORT → fragment linker           │  ~50 min
        │  → torso appearance measurement → FEATURE CACHE (JSONL)      │  (the model
        │  no counts are produced here                                 │   runs once)
        └──────────────────────────────────────────────────────────────┘
                                   │
                    staff/customer verdict, once per track,
                       from the COMPLETE evidence
                                   │
        ┌──────────────────────────────────────────────────────────────┐
 pass 2 │  replay cache → task state machines → CSVs                   │  ~5 min
        │  same objects → annotation → entrance_annotated.mp4          │  (decode only)
        └──────────────────────────────────────────────────────────────┘
                                   │
        ┌──────────────────────────────────────────────────────────────┐
 replay │  cache + a different config → every count, recomputed        │  ~3 s
        │  `rva.cli replay --set task3.staff.score_threshold=0.70`     │
        └──────────────────────────────────────────────────────────────┘
```

The second decode costs a few minutes against the ~50 that inference costs, and
it buys two things: the video and the CSVs are rendered from the same objects
and cannot disagree, and the cache makes every threshold in the project
re-testable in seconds (§4.4).

---

## 4. Methodology

### 4.1 The two ideas everything else is built on

**Scale-invariant "body-heights".** No camera calibration is available and both
lenses are wide-angle, so every distance that drives a decision is

```
        pixel distance between two things
norm =  --------------------------------------
        the person's own bounding-box height
```

A standing adult is ~1.7 m, so this reads directly as *distance in body
heights*. It is invariant to where in the frame the person stands, degrades
gracefully under lens distortion because numerator and denominator are measured
in the same neighbourhood, and it makes thresholds legible: `max_norm_dist: 1.0`
means "within one body-height of the shelf".

**Attention = fused body orientation, never gaze.** At 60–110 px a face model
has nothing to work with. Two independent estimates are fused
(`src/rva/core/orientation.py`):

- *Pose normal* — the perpendicular to the shoulder line. With
  `s = right_shoulder − left_shoulder` the facing vector is `(s.y, −s.x)`
  normalised; the two-fold sign ambiguity is resolved from image-space shoulder
  order (for someone facing the camera, the **left** shoulder appears on the
  image **right**). A head-yaw correction of up to ±55° is then applied from the
  nose offset relative to the shoulder mid-point, which recovers "walking one
  way while glancing sideways at the window".
- *Motion direction* — a walking person faces where they are going. Reliable
  when they are small, meaningless when they are still.

They are blended with a weight driven by normalised speed (`motion_bias: 0.65`),
and the **pose-only** estimate is also kept separately, because "turning their
head toward the store while walking past" is a different behaviour from
"turning their body", and the fused vector would erase the former.

### 4.2 One state machine for every count

All three tasks reduce to: *a condition is true on some frames and false on
others — how many distinct events happened?* `EpisodeTracker`
(`src/rva/core/events.py`) is the shared answer, with three named parameters:

| parameter | meaning |
|---|---|
| `min_duration_s` | the condition must hold this long before it is believed at all |
| `flicker_tolerance_s` | a shorter gap does **not** end the episode (missed detection, brief occlusion, a glance away) |
| `min_gap_s` | after an episode ends, re-engagement within this window is **merged back into it**; re-engagement after it counts as a **new** event |

`min_gap_s` *is* the answer to "how do you distinguish a continuous interest
episode from a later, separate return", and it is the same answer for Task 2's
shelf visits and Task 3's interaction sessions.

### 4.3 Protecting "unique person" counts from ID switches

Every headline number counts unique people, so an identity switch inflates it.
Three defences:

1. tracks shorter than `min_track_seconds` never contribute;
2. **`TrackLinker`** stitches a newly-born track onto a recently-dead one when
   the time gap (≤ 2.0 s), the spatial gap (≤ 1.8 body-heights) **and** the
   torso H–S colour-histogram correlation (≥ 0.60) all agree;
3. Task 1 **latches** its verdicts, so a count can never oscillate.

The linker is deliberately biased *against* merging: over-counting by one is
recoverable and visible, whereas wrongly merging two people silently corrupts
several metrics at once.

### 4.4 Threshold sensitivity, measured rather than asserted

Because the feature cache decouples inference from every decision rule, the
effect of each threshold on the headline numbers can be *measured*. These were
produced by replaying the submitted `entrance` cache, one `--set` at a time,
in about three seconds each:

**Task 1 — interest threshold** (`task1.interest.score_threshold`)

| value | Total Interested | Entered | Passed By |
|---:|---:|---:|---:|
| 1.00 | 12 | 2 | 10 |
| 1.10 | 12 | 2 | 10 |
| **1.20 (submitted)** | **6** | **2** | **4** |
| 1.30 | 4 | 1 | 3 |
| 1.45 | 3 | 1 | 2 |

**Task 1 — conversion rules.** `Interested Entered` is **2** for every entry
confirmation window from 0.6 s to 3.0 s, and for every entry depth from 0.25 to
0.70 body-heights. Conversion is a geometric event with a clean signature; the
pipeline finds the same people regardless of how those two knobs are set.

**Task 2 — shelf interest**

| variant | A | B | C | D | total |
|---|---:|---:|---:|---:|---:|
| `score_threshold` 0.70 | 7 | 5 | 4 | 4 | 20 |
| **submitted (0.90 / dwell 2.5 s / gap 6 s)** | **6** | **4** | **2** | **4** | **16** |
| `score_threshold` 1.10 | 5 | 5 | 2 | 4 | 16 |
| `min_dwell_s` 1.5 | 7 | 6 | 4 | 4 | 21 |
| `min_dwell_s` 4.0 | 4 | 4 | 1 | 3 | 12 |
| `min_gap_between_events_s` 3.0 | 8 | 4 | 3 | 4 | 19 |
| `min_gap_between_events_s` 12.0 | 6 | 4 | 2 | 4 | 16 |

**Task 3 — staff threshold** (`task3.staff.score_threshold`)

| value | Staff instances | Sessions | Average |
|---:|---:|---:|---:|
| 0.55 | 6 | 19 | 3.17 |
| 0.58 | 5 | 5 | 1.00 |
| **0.62 (submitted)** | **5** | **5** | **1.00** |
| 0.66 | 3 | 6 | 2.00 |
| 0.70 | 2 | 1 | 0.50 |

What this says honestly:

- **Conversion is robust.** `Interested Entered` does not move at all across the
  entry rules, and only drops when the *interest* threshold gets high enough to
  disqualify a person who did in fact walk in.
- **`Total Interested` is soft, because the underlying concept is.** It halves
  between 1.10 and 1.20. The value was chosen from the worked examples in §5 —
  what behaviour *should* qualify — not by fitting to a desired answer, and the
  table shows exactly what a different judgement would have produced.
- **Task 2's per-shelf ranking is stable**: A is the busiest and C the quietest
  under every setting tried, while the absolute totals move between 12 and 21.
  For a retail question ("which shelf draws attention?") the ranking is the
  answer, and it is the part that holds.
- **Task 3 is the most sensitive metric in the project**, because it divides two
  small numbers that both depend on the staff classifier. At 0.55 one further
  long-lived lounge track is admitted as staff and the session count more than
  triples. This is repeated in the limitations, and it is the single thing a
  labelled ground truth would fix (§10).

---

## 5. Task 1 — Store interest and conversion

> `src/rva/tasks/task1_interest.py`, `configs/entrance.yaml → task1`

### Who is a candidate

Only **passers-by**: a track whose *first* in-scene observation is on the
walkway. A track that begins already inside the store belongs to someone who
was shopping before the clip started, and counting them would inflate both
"interested" and "entered". (Not hypothetical — an early draft counted three
such people as conversions in the first 40 seconds.) They are still tracked and
still available to Task 3. Detected **staff are excluded** as well; they loiter
at the storefront all day.

People whose feet fall above `roi_top_y: 215`, or outside both the walkway and
store polygons, are on another mall level and are dropped entirely.

### Definition of interest

The brief says interest must **not** be reduced to "the person stopped", and
names four observable behaviours. Each is measured, normalised to 0–1, and
combined as a weighted sum:

| signal | measurement | weight |
|---|---|---|
| `proximity` | `1 − d/2.0`, where `d` = distance to the storefront line in body-heights. Also a **hard gate**: beyond 2.0 body-heights the person is on the far side of the walkway and is not a candidate at all. | 0.60 |
| `facing` | `1 − θ/70°`, where θ is the angle between the facing vector and the direction to the nearest point of the storefront — using **the better of the fused and the pose-only estimate**, scaled by keypoint confidence. "Looking toward the storefront, turning their head **or** body toward it." | **1.20** |
| `slow` | `1 − v/0.55`, where v is normalised speed. Note this is *slowing*, not stopping: 0.55 body-heights/s is roughly half a walking pace. | 0.60 |
| `approach` | ramps from 0 at a closing rate of 0.05 body-heights/s to 1.0 at 0.8 — "approaching the entrance", where the upper figure means *walking purposefully at the door* rather than merely drifting inward. | 0.90 |

```
interest_score = 0.60·proximity + 1.20·facing + 0.60·slow + 0.90·approach
```

**Threshold 1.20.** Worked examples:

| behaviour | score | verdict |
|---|---|---|
| looking at the store from ~1 body-height | 0.96 + 0.30 = **1.26** | interested |
| walking straight at the entrance, not slowing | 0.90 + 0.30 = **1.20** | interested |
| slowing near the store but looking away | 0.54 + 0.30 = **0.84** | not interested |
| merely walking close by | 0.30 = **0.30** | not interested |

No single behaviour is necessary or sufficient, which is exactly what the brief
asks for. The score must stay above threshold for **`min_evidence_s: 0.8`**
(bridging gaps up to 0.5 s) before the person is latched — one noisy frame
cannot mint a customer. Interest is **latched**: once shown it is never
withdrawn, so *Total Interested* is a count of unique people.

### Entered vs passed by

**ENTERED** — the feet are inside the store polygon, at least
`min_depth_norm: 0.45` body-heights past the storefront line, continuously for
`confirm_s: 1.2` seconds. The depth requirement implements *"crosses the
entrance boundary **and continues into the store**"*; the dwell requirement
rejects someone who leans in to look at a display and steps back out.

**PASSED_BY** — an interested person who leaves both zones for
`passby_confirm_s: 1.0` seconds without ever satisfying the entry condition. An
interested person still unresolved when the clip ends is attributed to
*passed by* (documented assumption: no entry was observed within the window).

**Interest must come first.** The brief asks for "those interested people who
**subsequently** entered", so once entry is confirmed the interest score is
frozen — wandering around inside the store afterwards cannot retro-fit the
evidence that justified the conversion. A person who walks in without ever
showing observable prior interest is therefore in *none* of the three counts;
they are flagged `ENTERED_NO_PRIOR_INTEREST` in the audit CSV rather than
silently dropped. One person in the submitted run falls into this category.

Every interested person receives exactly one outcome, so

```
Interested Entered + Interested Passed By == Total Interested
```

holds **by construction**. It is asserted in `pipeline.run_entrance` before any
CSV is written, and covered by a unit test.

---

## 6. Task 2 — Per-shelf customer interest

> `src/rva/tasks/task2_shelf.py`, `configs/interior.yaml → task2`

### Shelf assignment: exclusive, scored, with a reach tie-break

On every processed frame each customer is scored against all four shelves and
assigned to **at most one** — the highest scorer above threshold. Exclusive
assignment is what makes the totals meaningful: the same second of attention can
never be credited to two shelves.

| signal | measurement | weight |
|---|---|---|
| `proximity` | `1 − d/max_norm_dist`, where `d` is the distance from the **closest of the customer's feet and their hand** to the shelf fixture, in body-heights | 0.70 |
| `facing` | `1 − θ/75°` between the facing vector and the direction to the nearest point of the shelf | 0.70 |
| `reach` | `1 − φ/70°` between the shoulder→wrist vector and the direction to the shelf. Contributes only when a wrist keypoint is actually confident, so it never invents evidence. | **1.00** |

```
shelf_score = 0.70·proximity + 0.70·facing + 1.00·reach      threshold 0.90
```

Two design points worth defending:

- **Reach carries the most weight.** A hand in a shelf is the least ambiguous
  evidence available in this scene, and it is precisely what disambiguates the
  "standing in the gap between two shelves" case the brief calls out. Body
  orientation is noisy for a customer bent over a low gondola.
- **Proximity uses feet *or* hand, whichever is closer.** A shopper leaning over
  shelf B has their hand in the shelf while their feet are still in the aisle,
  and bounding boxes in this view are frequently truncated by the bottom frame
  edge — which makes the foot position the less honest of the two signals.

Per-shelf distance gates: A 1.1, B 1.0, C 1.1, D 1.0 body-heights.
Ground points above the ROI line (y = 230) are the service counter on the far
side of the store and are excluded.

### Continuous episode vs a genuine return

Assignment feeds an `EpisodeTracker` per (customer, shelf) pair:

| parameter | value | rationale |
|---|---|---|
| `min_dwell_s` | 2.5 s | attention must be sustained — walking past a shelf is not an event |
| `flicker_tolerance_s` | 1.5 s | a shorter break (missed detection, glance away, occlusion behind a gondola) does not end the episode |
| `min_gap_between_events_s` | 6.0 s | re-engagement within 6 s is merged back into the previous episode; after 6 s it is a **new** event |

So a customer who browses one shelf continuously for 45 seconds scores **one**
event, and the same customer returning ten seconds later scores a **second** —
which is exactly the behaviour the brief specifies. Both cases are unit-tested.

### On-screen (all four requirements from the brief)

- **which shelf** — the person's box, the connecting line and the label all take
  that shelf's colour;
- **duration** — the live elapsed time of the current episode, on the link;
- **association** — a line from the customer to the nearest point of the shelf;
- **cumulative counts** — a chip on every shelf plus the top-left panel.

---

## 7. Task 3 — Staff–customer interaction

> `src/rva/tasks/task3_staff.py`, `src/rva/core/staff.py`, `configs/entrance.yaml → task3`

### Identifying staff

Three cheap, independent, explainable signals per track. None is trustworthy
alone, so the classifier only commits after `min_obs: 12` torso observations,
and the verdict is then **latched** so the Task 3 denominator cannot drift.

| signal | how it is measured |
|---|---|
| `apron` | fraction of torso pixels that are dark **and** desaturated (V ≤ 95, S ≤ 120) |
| `badge` | **a shape test.** A connected bright component (V ≥ 140) covering 0.5–6 % of the torso, aspect ≥ 0.60, fill ≥ 0.42, sitting on the dark field. Scored by **detection rate** over frames where the person is ≥ 110 px tall (the badge is ~15 px across at that size; below it the test is meaningless). Seeing it on ≥ 25 % of resolvable frames scores full marks. |
| `residency` | seconds spent inside the store polygon, saturating at 30 s. Customers transit; staff stay. |

```
staff_score = 0.35·apron + 0.45·badge + 0.20·residency        threshold 0.62
```

The badge parameters were **measured on real crops from this clip**, not
guessed: at t = 10 s the badge is a 20×17 px blob covering 3.6 % of the torso
with aspect 0.85 and fill 0.50. The scene is dim — the 99th percentile of torso
brightness is only ~156 — which is why the bright threshold is 140 rather than
the 200+ a well-lit scene would use. The badge carries a dark logo, so its
bright mask is a ring (fill ≈ 0.50) rather than a disc (0.79).

Why the badge, and not just "dark clothing": measured dark-torso ratios on
genuine shoppers in this footage run **0.83–0.96**, higher than on the actual
apron. Appearance darkness alone would misclassify half the customers.

Worked examples against the 0.62 threshold:

| person | score | verdict |
|---|---|---|
| staff: apron 0.55, badge rate 0.5, resident | 0.19 + 0.45 + 0.20 = **0.84** | staff |
| dark-clothed customer, whole clip, no badge | 0.32 + 0.00 + 0.20 = **0.52** | customer |
| same customer with a 5 % badge false-positive rate | 0.32 + 0.09 + 0.20 = **0.61** | customer |

A **staff instance** is one continuous appearance in view — exactly one linked
track id, which is what the brief asks for ("treat as the same staff instance
for as long as they remain within the camera view; re-identification across
separate appearances is optional"). A "staff instance" that flickered for under
2 s and served nobody is dropped as a detection artefact, since including it
would deflate the average with a spurious zero.

`outputs/audit/task3_staff_features.csv` dumps every feature for every track, so
the classification can be audited or the thresholds re-tuned without re-running
inference blind. `force_staff_ids` / `force_customer_ids` in the config are
reproducible manual overrides; **both are empty in the submitted run** — the
reported numbers are fully automatic.

### What counts as an interaction

Being near each other in a small shop is not engagement. Two routes qualify:

**Engaged-facing** — within `max_norm_dist: 1.9` body-heights **and** at least
one of the pair oriented towards the other within `facing_angle_deg: 80`.
Requiring only one direction is deliberate: in a shoe fitting the staff member
faces the customer while the customer looks down at the shoe, so a mutual-gaze
rule would miss the single most common interaction in this footage.
(`require_mutual: false` — set it to `true` to tighten.)

**Co-stationary** — within 1.5 body-heights **and** both essentially stationary
(≤ 0.22 body-heights/s). This is the fallback for the seating area at the bottom
of the frame, where the staff member kneels, is truncated by the frame edge and
occluded by the armchair, and pose is unreliable. Without it the pipeline would
systematically under-count exactly the interactions the store cares most about.

### Sessions and the average

Per (staff, customer) pair the engagement flag feeds an `EpisodeTracker`:

| parameter | value | rationale |
|---|---|---|
| `min_session_s` | 2.0 s | a customer walking past a staff member is not a session |
| `flicker_tolerance_s` | 1.0 s | brief detection dropouts do not split a session |
| `end_gap_s` | 3.0 s | engagement stopping for longer closes the session; later re-engagement between the same pair is a **new** session |

Because sessions are per *pair*, one staff member serving two customers at once
accrues two sessions, and the same customer returning later accrues another —
which is the "sessions, not unique customers" requirement.

```
average = total interaction sessions / number of staff instances
```

with **every** staff instance in the denominator, including those with zero
sessions. The brief's worked example — (5 + 4 + 0) / 3 = 3.0 — is a unit test.

---

## 8. Outputs

```
outputs/
├── entrance_annotated.mp4          Task 1 + Task 3 (one video, as required)
├── interior_annotated.mp4          Task 2
├── task1_store_interest.csv        Total Interested / Entered / Passed By
├── task2_shelf_interest.csv        events per shelf A-D, plus TOTAL
├── task3_staff_interactions.csv    per staff instance + the AVERAGE row
├── cache/                          feature cache (only with --dump-features)
└── audit/                          not required by the brief - provided so the
    ├── task1_per_person.csv        numbers can be traced to the second
    ├── task1_events.jsonl
    ├── task2_episodes.csv
    ├── task2_events.jsonl
    ├── task2_staff_features.csv
    ├── task3_sessions.csv
    ├── task3_staff_features.csv
    ├── task3_events.jsonl
    ├── entrance_summary.json       results + the fully resolved config
    └── interior_summary.json
```

Every increment of every headline number appears in the JSONL event log with its
timestamp and the evidence that triggered it, and every episode appears in the
audit CSVs with its start, end, duration and how many gaps were bridged. **CSV
values and on-screen counters are produced from the same objects**, so they
cannot disagree.

### Reading the annotated videos

The metrics panel sits at the top-left, offset below the camera's own burned-in
timestamp. Colour encodes *state*, not identity:

| colour | meaning |
|---|---|
| amber | interested (Task 1) |
| green | interested → entered |
| salmon | interested → passed by |
| gold | staff instance |
| cyan link | an active staff–customer interaction session, with its live duration |
| shelf colours | which shelf a customer is engaged with (Task 2) |
| grey | a tracked person not currently contributing to any count |

Zone outlines (walkway, store, entrance boundary, shelf polygons) are drawn so a
reviewer can see the geometry every decision was made against. Pose skeletons
are drawn because they are the evidence the orientation and reach terms use —
`--no-zones` and `output.draw_keypoints: false` turn them off.

---

## 9. Verification

```bash
make test          # 74 unit tests, ~1 s, no model or video needed
```

The tests are executable statements of the methodology rather than coverage
filler:

- *walking past a shelf is not an interest event*
- *a 45-second continuous visit is one event, not 45*
- *a return after a long absence is a second event*
- *a brief turn away does not split an episode*
- *leaning in over the threshold and stepping back is not an entry*
- *someone already inside the store is not a passer-by*
- *the same attention is never credited to two shelves*
- *zero-interaction staff are included in the average — (5+4+0)/3 = 3.0*
- *entered + passed_by == total_interested*
- *unique-person latches never flip back*
- *the shipped zone polygons are valid and the storefront line separates them*

Beyond the unit tests: the Task 1 invariant is asserted at runtime before the
CSVs are written; the zone configuration is verifiable visually with
`rva.cli zones`; the audit trail lets any reported number be checked against the
timestamped source footage; and the feature cache lets any threshold's effect on
the headline numbers be measured in seconds rather than argued about.

---

## 10. Assumptions and limitations

### Assumptions

1. **Fixed camera.** Zone polygons are in image coordinates. If either camera
   moves, the configs must be redrawn (`rva.cli grid` + `rva.cli zones`).
2. **Body height as a scale reference.** Standing adults dominate the footage;
   a seated or crouching person's bounding box under-states their scale, which
   inflates their normalised distances slightly.
3. **Each clip is evaluated independently**, as the brief permits — no identity
   persistence across videos.
4. **Interested people unresolved at the end of the clip are counted as passed
   by.** They were never observed entering within the window.
5. **A track beginning inside the store is not a passer-by** (Task 1 only).
6. **One linked track = one staff instance**, per the brief's definition.
7. **`interior.mp4` is treated as all-customer.** See below.

### Known limitations and failure cases

| # | Limitation | Effect | Mitigation in place |
|---|---|---|---|
| 1 | **No staff exclusion in the interior view.** The apron signal does not survive that camera: it is further away and from above, several genuine customers score dark-torso ratios of 0.83–0.96, and no badge is resolvable. | If a staff member browses a shelf it is counted as customer interest. | Deliberately left off rather than excluding people on an unreliable signal (`task2.exclude_staff: false`, with the reasoning in the config). The classifier still runs and its evidence is written to `audit/task2_staff_features.csv`. |
| 2 | **Truncated bounding boxes** at the bottom frame edge under-state a person's height and misplace their feet. | Slightly inflated normalised distances in the lounge/fitting area. | Task 2 measures proximity from the hand as well as the feet; Task 3 has the co-stationary fallback that does not depend on pose. |
| 3 | **Identity fragmentation** behind fixtures remains the largest single error source for every "unique person" count. | Over-counting. | Short-track filter, colour-histogram fragment linker, latched verdicts. The linker is intentionally conservative. |
| 4 | **Orientation is body orientation, not gaze.** A person who looks at the store using only their eyes will be missed. | Under-counting interest. | The head-yaw correction recovers head turns; eye-level gaze is not recoverable at 60–110 px and is not claimed. |
| 5 | **Lens distortion is not corrected.** | Small systematic error in normalised distances near the frame edges. | Body-height normalisation is local, so the error largely cancels; a homography was rejected for lack of any calibration reference. |
| 6 | **Low traffic amplifies single errors.** With counts in the single digits, one false trigger is a large relative change. | Sensitivity to thresholds. | Sustained-evidence rules everywhere; every threshold is exposed and documented so a reviewer can test the sensitivity directly with `--set`. |
| 7 | **`frame_stride: 3`** means analysis runs at 10 Hz. | An event shorter than ~0.3 s could be missed. | All rules require ≥ 0.8 s of evidence, so this is well inside the margin. Set `runtime.frame_stride: 1` for full rate. |
| 8 | **Reflections in the balustrade glass** can produce phantom detections. | Potential false passers-by. | The ROI cut and walkway polygon suppress the region; the empirical scan found essentially no detections above y = 225. |
| 9 | **The "co-stationary" interaction rule** could in principle fire on two customers standing together, if one were misclassified as staff. | Inflated session count. | It requires one party to already be classified as staff, which needs badge evidence. The measured consequence is visible in §4.4: dropping the staff threshold to 0.55 admits one extra lounge track and triples the session count. |
| 10 | **Staff who are only ever seen kneeling or truncated are missed.** A fitting at the very bottom of the entrance frame gives few clean torso observations, so the classifier never reaches `min_obs` on that person. | Under-counted staff instances and their sessions. | Deliberate: the alternative is to lower `min_obs`, which a first pass showed produces 19 "staff" from 93 tracks. Precision was preferred over recall, and the trade-off is measurable via the cache. |
| 12 | **A person can enter without registering prior interest** — walking straight in at an angle where neither the facing nor the approach signal fires. | Under-counted conversions. | Reported explicitly as `ENTERED_NO_PRIOR_INTEREST` in `audit/task1_per_person.csv` (1 case in the submitted run), so the gap is visible rather than hidden. |
| 11 | **Two passes over the video.** Deciding staff roles from whole-clip evidence means the video is decoded twice. | ~10 % more wall time. | Accepted: inference still runs only once, and the alternative was a role that changed mid-clip, which would break the CSV-matches-video requirement. |

### What I would do next with more time

1. **A small apron classifier.** A few hundred torso crops labelled from these
   two clips would train a linear probe on CLIP or MobileNet features and
   replace the hand-built colour/shape heuristic — the single biggest accuracy
   win available, and it would make interior staff exclusion viable.
2. **Appearance-based ReID in the tracker** (BoT-SORT with a ReID backbone) to
   cut identity fragmentation at source.
3. **A hand-annotated ground truth** for both clips — even 20 events — to turn
   "the methodology is consistent" into a measured precision/recall figure and
   let the thresholds be tuned against something other than judgement. The
   feature cache makes this cheap: a sweep over any threshold is seconds of
   compute, so the only missing ingredient is labels.
4. **Camera calibration** from the known floor-tile pitch in the mall walkway,
   which would give genuine metric distances and let the thresholds be stated
   in metres.

---

## 11. Repository structure

```
.
├── README.md                     this document
├── ASSIGNMENT_UNDERSTANDING.md   brief analysis, footage analysis, delivery plan
├── Dockerfile                    reproducible CPU environment (runs the tests at build)
├── Makefile                      make setup / test / run / smoke / zones
├── requirements.txt              pinned dependencies
├── pytest.ini
├── configs/
│   ├── common.yaml               model, tracker, runtime, fragment linker
│   ├── botsort_rva.yaml          BoT-SORT tuned for a fixed camera
│   ├── entrance.yaml             entrance zones + Task 1 and Task 3 parameters
│   └── interior.yaml             shelf polygons + Task 2 parameters
├── docs/                         calibration evidence (traffic plots, zone
│                                 previews, badge-detector check)
├── models/
│   └── download_models.sh        fetches the public YOLO11-pose weights
├── tools/
│   └── compress_outputs.sh       optional H.264 re-encode of the annotated videos
├── src/rva/
│   ├── cli.py                    command line entry point
│   ├── config.py                 YAML loading, inheritance, --set overrides
│   ├── pipeline.py               the two runnable pipelines + the annotation layer
│   ├── core/
│   │   ├── geometry.py           polygons, polylines, body-height normalisation
│   │   ├── replay.py             feature cache: re-tune thresholds without inference
│   │   ├── orientation.py        pose + motion orientation fusion, reach vector
│   │   ├── events.py             EpisodeTracker and Latch - every count flows here
│   │   ├── tracking.py           detector, tracker, fragment linker, track state
│   │   ├── staff.py              apron + badge + residency classifier
│   │   ├── viz.py                overlays, panels, links, legends
│   │   ├── video.py              frame reader/writer
│   │   └── report.py             CSV / JSONL / JSON writers
│   └── tasks/
│       ├── task1_interest.py     store interest and conversion
│       ├── task2_shelf.py        per-shelf customer interest
│       └── task3_staff.py        staff-customer interaction
├── tests/                        74 unit tests
├── data/                         entrance.mp4, interior.mp4 (not committed)
└── outputs/                      generated deliverables
```

Every tunable number lives in `configs/`, with a comment stating its rationale
next to it. Nothing that can change a reported count is hard-coded in the source.
