# Hendricks CV Take-Home — Understanding, Footage Analysis & Delivery Plan

> Working document. This is the *thinking* behind the submission: what the
> assessment actually asks for, what is genuinely in the two videos, the
> decisions those two things force, and the step-by-step plan to deliver.
> `README.md` is the reviewer-facing document.

---

## 1. What the assessment is really testing

Read the rubric before the task list, because the weights invert the obvious
reading of the brief:

| Criterion | Weight | What it means in practice |
|---|---:|---|
| Task correctness & completeness | 35% | Every output exists, is internally consistent, and follows from the stated method. **"Perfect model-level accuracy is not expected."** |
| Technical reasoning & methodology | **45%** | The event definitions, temporal rules, thresholds, formulas and *justifications* — demonstrated **through the documentation**. |
| Presentation, communication, visualisation | 20% | Code quality, README clarity, readability of the annotated videos. |

Two consequences drive every decision in this repo:

1. **The README is worth more than the model.** 45% is awarded for reasoning
   that is *written down*. A pipeline with mediocre recall and a rigorous,
   defended methodology scores far better than a strong detector with hand-
   wavy rules. Every threshold therefore lives in a commented YAML file and
   has a stated rationale.
2. **§6.3: a candidate can be rejected for not being able to defend the
   submission, regardless of score.** So: no black boxes, no magic numbers,
   no library whose behaviour cannot be explained. Every decision must be
   reducible to one sentence in an interview.

### The three tasks

| | Source | Metric | Required outputs |
|---|---|---|---|
| **Task 1** | `entrance.mp4` | Total Interested / Interested Entered / Interested Passed By | overlay counts + CSV |
| **Task 2** | `interior.mp4` | Interest events per shelf A–D | overlay showing shelf assignment, live duration, cumulative counts + CSV |
| **Task 3** | `entrance.mp4` | Average interaction **sessions** per staff instance | **same video as Task 1** + CSV |

Only **two** annotated videos in total — Task 3 must share the entrance video
with Task 1. That is stated twice in the brief and is an easy way to lose
correctness marks.

### Hard requirements checklist

- [x] Counts displayed **top-left** of the output video
- [x] All ML detections visibly annotated
- [x] Interaction results presented so the interaction is understandable
- [x] Every criterion/threshold has a comment near the relevant code
- [x] CSV values match the video overlays exactly
- [x] Staff instances with **zero** interactions included in the Task 3 average
- [x] No external/paid inference APIs; nothing uploaded anywhere
- [x] Dockerfile + dependency spec + `src/` + `configs/` + `outputs/` + `models/` or download instructions
- [x] README documenting methodology, thresholds, formulas, setup, assumptions, limitations

### The shape shared by all three tasks

Every task is the same problem wearing a different hat:

> Turn a fuzzy human notion ("interest", "engagement") into a **per-frame
> boolean** over tracked identities, then convert that noisy boolean into a
> **count of distinct events** using explicit temporal rules.

That observation is the architectural backbone: one tracker, one orientation
estimator, one hysteresis state machine (`EpisodeTracker`), three thin task
layers on top. Anything else duplicates the hardest logic three times and
guarantees the three answers are justified inconsistently.

---

## 2. What is actually in the footage

Both clips are 1280×720, 30 fps, MPEG-4, **≈7.5 minutes** each (not the
"snippet" length one might assume): `entrance.mp4` is 447 s / 13,421 frames,
`interior.mp4` is 431 s / 12,947 frames. Burned-in timestamps show they are
near-simultaneous on 2023-11-07: entrance starts 15:27:13, interior 15:27:19.
It is a shoe store in an Australian shopping centre (Myer bags, orange
shoeboxes).

### 2.1 `entrance.mp4` — the camera is *inside* the store looking out

This is the single most important fact about the clip and it inverts the
naive reading of "storefront camera". The frame is three horizontal bands:

```
  y≈0–215    ATRIUM VOID — glass balustrade, escalator, and the mall level
             BELOW visible through the glass (Specsavers, NYX, Camera House).
             People here are on a different floor. MUST BE EXCLUDED.
  y≈210–455  MALL WALKWAY — grey tile with dark diamond inserts. The
             passers-by Task 1 is about.
  y≈290–720  STORE INTERIOR — cream tile and navy carpet, armchair lounge,
             display plinths, shoe walls left and right.
```

Consequences that shaped the implementation:

- **There is no door.** The frontage is open, so "the entrance boundary" is a
  *material transition* in the floor — the lease line — which had to be drawn
  by hand as a polyline (`scene.storefront_line`).
- **The upper band must be masked.** A `roi_top_y` cut plus walkway
  containment removes another floor's traffic from the counts.
- **Passers-by are 60–110 px tall.** Gaze estimation is hopeless at that
  size; the head is a dozen pixels and often seen from above or behind. Body
  orientation from the shoulder line, fused with motion direction, is the
  only workable "attention" signal. Measured detection heights over the whole
  clip: p1 = 64 px, median = 138 px, p99 = 224 px.
- **Traffic is light** — typically 0–3 people in the walkway at a time. The
  counts will be small, so one false trigger moves the metric materially.
  This argues for *conservative* thresholds and sustained-evidence rules.
- **Glass and mirrors on the right** produce reflections; the walkway polygon
  and ROI cut suppress most of the resulting phantom detections.
- **Staff are legible.** The uniform is a dark navy bib apron with a small
  circular light badge on the chest, and staff work close to the camera
  (120–250 px tall), so a torso-crop colour statistic is viable. But the
  store's armchairs and carpet are *the same navy*, so appearance alone is
  not enough — a behavioural prior (residency inside the store) is needed.
- **The hard case for Task 3 is at the bottom edge.** Shoe fittings happen in
  the armchair lounge at the very bottom of frame, where the staff member
  kneels, is truncated by the frame border and occluded by the chair. Pose
  keypoints there are unreliable. Any interaction rule that *requires* pose
  will systematically miss the store's most important interaction — hence the
  co-stationary fallback.

**Empirical calibration.** Rather than eyeballing the zone polygons, the
detector was run over the entire clip at 2 Hz (3,017 detections) and every
foot position plotted onto a reference frame. Two things fell out:
essentially **no** detections above y≈225 (confirming the ROI cut is safe and
that other-floor people are largely not detected anyway), and a clear dead
band between the walkway traffic and the in-store traffic, caused by the
store's front-line fixtures. The storefront line was drawn **through that
dead band**, which makes the exact placement far less sensitive than it looks.

### 2.2 `interior.mp4` — high corner camera, strong barrel distortion

Straight shelf edges visibly bow. Mapping the brief's Figure 1 label anchors
onto the footage:

| Shelf | Fixture in the footage |
|---|---|
| **A** | tall gondola on the right, dark shoes + the colour sock rack |
| **B** | low wooden gondola front-centre, orange shoebox rack at its end |
| **C** | long tiered display running up-right across the middle |
| **D** | shoe wall down the left-hand side |

The brief gives *label anchors*, not polygons — the regions are ours to
define and defend.

Consequences:

- **Lens distortion rules out a single ground-plane homography.** With no
  known floor references and visible barrel distortion, a planar homography
  would be wrong at the edges. Every distance is therefore normalised by the
  person's own pixel height (see §3.1).
- **Occlusion is constant.** Shelves cut people off at the waist; customers
  disappear behind gondolas for a second or two. This is exactly what
  `flicker_tolerance_s` and the fragment linker exist for.
- **The between-shelves case is real and frequent.** People genuinely stand
  in the aisle between B and A, and between B and D. The brief calls this out
  explicitly, so the tie-break has to be principled: exclusive assignment to
  the highest-scoring shelf, with **reach direction** as the decisive term.
- **The double-count trap is present in the first minute.** One customer
  (blonde, olive dress) dwells at shelf B for ~30 s continuously and returns
  later — precisely the "continuous episode vs. separate return" distinction
  the brief asks us to define.
- **The far band is a different activity.** The empirical scan shows a dense
  cluster at y≈130–230 around x≈760–1060: people at the service counter on
  the far side of the store. They are not engaging with A–D, so the ROI is
  cut below them.
- **Overhead angle means top-of-head views**, so shoulder-line orientation
  again beats head pose.

### 2.3 One shared nuisance

The source video **already burns a timestamp into the top-left corner** —
exactly where the brief wants the counts. The metrics panel is therefore
offset to start below it rather than fighting with it.

---

## 3. The decisions this forces

### 3.1 Scale-invariant units: "body-heights"

No metric calibration is available, so every distance that drives a decision
is `pixel_distance / person_pixel_height`. A standing adult is ~1.7 m, so
this reads as "distance in body-heights", it is invariant to where in the
frame the person stands, and it degrades gracefully under lens distortion
because numerator and denominator are measured in the same neighbourhood.
It also makes thresholds *legible*: `max_norm_dist: 1.0` means "within one
body-height of the shelf", which is defensible in an interview without
reference to pixels.

### 3.2 One pose model, not a detector plus a pose network

YOLO11-pose returns the box **and** 17 COCO keypoints in one forward pass.
Keypoints are needed three times over — body orientation, reach direction,
and the torso crop for the staff classifier — so a pose model is strictly
cheaper than two models on a CPU-only budget, and box↔keypoint association
comes for free.

### 3.3 Attention = fused body orientation, never gaze

Two independent estimates, blended by normalised speed:

- **Pose normal.** The perpendicular to the shoulder line. The two-fold sign
  ambiguity is resolved from image-space shoulder order (for someone facing
  the camera, the *left* shoulder appears on the image *right*). Head yaw is
  added as a correction from the nose offset, which recovers "walking one way
  while looking sideways at the window".
- **Motion direction.** A walking person faces where they are going —
  reliable when they are small, meaningless when they are still.

Fast movers trust motion; near-stationary people trust pose.

### 3.4 One hysteresis state machine for all three counts

`EpisodeTracker` with three named parameters:

- `min_duration_s` — evidence must be sustained before it counts at all;
- `flicker_tolerance_s` — a shorter break does **not** end the episode;
- `min_gap_s` — after an episode ends, re-engagement within this window is
  merged back in; re-engagement after it is a **new** event.

That third parameter *is* the answer to "how do you distinguish a continuous
interest episode from a later, separate return", and it is the same answer
for Task 2's shelf visits and Task 3's interaction sessions. Consistency
across tasks is itself a reasoning mark.

### 3.5 Guarding "unique people" against ID switches

Every headline number is a count of unique people, so an identity switch
inflates it. Three defences: short tracks are ignored; a conservative
fragment linker stitches a new track onto a recently-dead one when time,
position **and** torso colour histogram all agree; and Task 1 latches its
interest verdict so a count can never oscillate. The linker is deliberately
biased towards *not* merging — over-counting by one is recoverable, silently
merging two different people corrupts several metrics at once.

### 3.6 Who counts as a passer-by

A track whose first in-scene observation is already inside the store belongs
to someone shopping before the clip began. They are not a passer-by, and
counting them inflates both "interested" and "entered". Task 1 therefore
requires **walkway origin**. (This was not a hypothetical: the first draft
counted three such people as conversions in the first 40 seconds.)

---

## 4. Delivery plan — step by step

**Phase 0 — Read the brief against the rubric.** ✅ Done above. The output is
the requirements checklist in §1 and the realisation that the README carries
45%.

**Phase 1 — Understand the footage before writing any code.** ✅
Sample both clips, identify the camera geometry, measure detection heights,
locate the shelves against Figure 1, and find the failure modes (other mall
levels, reflections, frame-edge fittings, between-shelf ambiguity). Every
architectural decision in §3 came out of this phase; doing it after writing
code would have meant rewriting the code.

**Phase 2 — Calibrate zones empirically, not by eye.** ✅
Run the detector over both full clips at 2 Hz, plot every foot position,
then draw the walkway / store / storefront-line and the four shelf polygons
over the actual traffic. Render an overlay preview and verify visually.
Deliverable: `configs/entrance.yaml`, `configs/interior.yaml`, and the
zone-preview images.

**Phase 3 — Build the scene-agnostic core.** ✅
`geometry` (polygons, polylines, body-height normalisation) → `orientation`
(pose + motion fusion) → `events` (the hysteresis machine) → `tracking`
(detector, tracker, fragment linker, per-track derived signals) →
`staff` (apron + badge + residency classifier) → `viz` / `report`.

**Phase 4 — Build the three task layers on top.** ✅
Each is a thin scorer plus state machines: Task 1 (four-signal interest score
→ latch → entry/pass-by), Task 2 (exclusive shelf assignment → per-pair
episodes), Task 3 (pairwise engagement → per-pair sessions → average).

**Phase 5 — Unit-test the counting logic against synthetic tracks.** ✅
73 tests. The important ones are executable statements of the methodology:
"walking past a shelf is not an event", "a 45-second continuous visit is one
event", "a return after a long absence is two", "leaning in and stepping back
is not an entry", "zero-interaction staff are included in the average",
"entered + passed_by == total_interested". These run in under a second with
no model and no video, so the logic can be iterated on instantly.

**Phase 6 — Smoke-run on 60–90 second slices.** ✅
Inspect annotated frames, read the per-person audit CSV, fix what the real
footage exposes that synthetic tracks cannot. Four things came out of this
phase that no amount of synthetic testing would have found:

* the **walkway-origin rule** — the first draft counted three people who were
  already inside the store as conversions in the first 40 seconds;
* the **co-stationary interaction fallback** — pose is unusable for the kneeling
  staff member at the bottom frame edge, which is where the fittings happen;
* the **badge shape test**, measured off real crops (dark-torso ratio alone
  scores 0.83–0.96 on ordinary shoppers, higher than on the actual apron);
* the **two-pass architecture** — a single pass has to commit to a staff verdict
  in a track's first seconds, and the verdict then changed mid-clip, leaving the
  video and the CSVs disagreeing. That is a correctness failure against the
  brief, so the pipeline was restructured: inference and evidence first, then
  one verdict per track from the whole clip, then metrics and annotation.

**Phase 6b — Build the feature cache.** ✅
Inference is ~50 minutes per clip and *nothing* about it depends on any
threshold. Caching the per-frame detections, keypoints and appearance
measurements turns a threshold change from an hour into three seconds, which is
what made the sensitivity table in README §4.4 possible at all.

**Phase 7 — Full runs on both clips.** ✅
~50 min of inference per video on 2 CPU cores at `frame_stride: 3`, plus ~5 min
to annotate. Produces the two annotated videos, the three required CSVs, the
audit trail and the feature caches.

Results on the supplied footage:

| | |
|---|---|
| Task 1 | 6 interested → 2 entered, 4 passed by (+1 entered with no prior interest, flagged in the audit CSV) |
| Task 2 | A 6, B 4, C 2, D 4 (16 events) |
| Task 3 | 5 staff instances, 5 sessions, **average 1.00** |

**Phase 8 — Verify.** ✅
Four independent checks:

1. **Replay reproduces the run.** Replaying each feature cache with the
   unchanged config returns exactly the numbers the full run reported.
2. **Events checked against the raw footage.** Every Task 1 event was located
   in the source video at its logged timestamp — e.g. the conversion at
   t = 334→339 s is a woman crossing from the walkway to the display and
   staying; the 20.8-second interest episode at t = 355 s is a shopper standing
   at the storefront who then leaves.
3. **The Task 1 invariant** (`entered + passed_by == total_interested`) is
   asserted in code before any CSV is written.
4. **Threshold sensitivity measured**, not asserted — see README §4.4. Every
   number in those tables came from a three-second cache replay.

**Phase 9 — Write the README as the primary deliverable.** ✅
Methodology per task, every threshold with its rationale, formulas, setup and
run commands, output inventory, assumptions, limitations and known failure
cases.

**Phase 10 — Package.** ✅
`Dockerfile` (tests run at build time), `requirements.txt`, `Makefile`,
`models/download_models.sh`, and the ZIP.

---

## 5. Rubric self-audit

| Requirement | Where it is satisfied |
|---|---|
| Three counts on `entrance.mp4`, top-left overlay + CSV | `pipeline._render_entrance`, `outputs/task1_store_interest.csv` |
| Interest defined by more than "stopped" | four weighted signals: proximity, facing, slowing, approach |
| Entered vs passed by, thresholds explained | depth + dwell confirmation; `task1.entry` in the config |
| Per-shelf events on `interior.mp4` + CSV | `outputs/task2_shelf_interest.csv` |
| Between-shelf ambiguity resolved | exclusive assignment, reach-direction tie-break |
| No double counting; return = new event | `EpisodeTracker` `flicker_tolerance_s` vs `min_gap_s` |
| Shelf, duration, association, cumulative counts on screen | `pipeline._render_interior` |
| Staff identified; same instance while in view | apron+badge+residency classifier; one linked track = one instance |
| Zero-interaction staff in the average | `StaffInteractionTask.average_sessions` + unit test |
| Sessions, not unique customers; returns count again | per-pair `EpisodeTracker` |
| Task 3 shares the Task 1 video | one `entrance_annotated.mp4` |
| Rationale comments near the code | every config block and module docstring |
| Reproducible environment | `Dockerfile`, pinned `requirements.txt`, weight download script |
| Assumptions, limitations, failure cases | README §"Assumptions and limitations" |
| Candidate can defend it | no black boxes; audit CSV/JSONL traces every increment to a timestamp |
