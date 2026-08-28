# Does this scale across stores?

> A standing answer to the question a reviewer is most likely to ask about a
> solution built against two clips from one store. `README.md` documents what
> was built; this documents how far it would travel and what would have to
> change first.

---

## Verdict

**The architecture scales. The configuration does not. As shipped, it would not
survive a second store without a person sitting down with each camera.**

That is not a hedge — it is a specific, testable claim, and the rest of this
document names exactly which parts fall on each side of the line. Roughly:

| | |
|---|---|
| **Portable today** | the units, the counting logic, the config layering, the compute profile, the re-tuning workflow |
| **Per-camera human work today** | zone geometry — the single biggest blocker |
| **Per-chain rewrite today** | the staff/uniform classifier |
| **Needs re-derivation per store *format*** | the behavioural definitions themselves |

---

## 1. What transfers unchanged

### 1.1 Body-height normalisation

The most portable decision in the project. Every distance that drives a
decision is

```
norm = pixel_distance / the person's own bounding-box height
```

which makes it invariant to camera height, focal length, lens, mounting angle
and where in the frame the person stands. A threshold expressed in pixels
transfers to *nothing*; `max_norm_dist: 1.0` — "within one body-height of the
shelf" — has a real chance of meaning the same thing in a store nobody has
looked at yet.

This is what makes fleet-wide threshold defaults conceivable at all. Without
it, every store would need not just new zones but new numbers.

### 1.2 Temporal rules in seconds, never frames

`min_dwell_s`, `flicker_tolerance_s`, `min_gap_s`, `confirm_s`,
`min_session_s` — all in seconds. A store with 15 fps cameras, or an edge box
that has to run at `frame_stride: 6` to keep up, produces the same counts. No
per-site frame-rate arithmetic, and no silent drift when hardware is replaced.

### 1.3 Config inheritance is already the fleet shape

```
configs/common.yaml      → chain-wide model, tracker, runtime defaults
configs/<scene>.yaml     → extends: common.yaml, then per-camera overrides
```

A new store is a new YAML file, not a fork of the code. `--set key=value`
overrides anything at run time without editing a file, which is what a
deployment tool would use. Nothing that can change a reported count lives in
Python.

### 1.4 One `EpisodeTracker` for every count

All three metrics reduce to *"a condition is true on some frames and false on
others — how many distinct things happened?"* A fourth metric — queue dwell,
fitting-room conversion, till abandonment — is a new **scorer**, not new
counting logic, and it inherits the same defensible answer to "when is this one
event and when is it two?"

### 1.5 CPU-only and fully local

No cloud inference, no footage leaving site. That removes the per-store egress
bill, the privacy review that comes with shipping customer video off-premises,
and the failure mode where a WAN outage stops analytics. It also means the same
code runs on an edge box and in a datacentre.

### 1.6 The feature cache is the sleeper scaling lever

Nothing in the inference pass depends on any threshold, so it is cached and
replayed. At fleet scale this stops being a convenience and becomes the
deployment mechanism:

- a proposed config change can be **validated against every store's cached
  history in seconds** before it is rolled out;
- a disputed number can be re-derived without touching the original footage;
- threshold fitting against labels becomes a sweep, not a re-run.

Re-tuning one threshold across 500 stores is minutes of compute instead of
500 hours.

---

## 2. What breaks, hardest first

### 2.1 Hand-drawn zone polygons — the blocker

`configs/entrance.yaml` and `configs/interior.yaml` carry, in **image pixel
coordinates for one specific camera**:

- `store_polygon`, `walkway_polygon`, `storefront_line`, `roi_top_y`
- four shelf fixture polygons with per-shelf distance gates

500 stores × 3 cameras is 1,500 manual calibrations. Worse, the coupling is
silent: a camera nudged by a cleaner, a fixture reset at a seasonal
changeover, or a lens knocked by 5° leaves a config that still *runs* and still
produces numbers — just wrong ones, with nothing in the output to say so.

Today, per camera, a human must: dump a coordinate-grid frame, run a detection
scan to see where traffic actually goes, draw between three and seven polygons,
render the overlay preview, eyeball it, run a 60-second smoke test, and sanity
check the resulting counts. Call it **30–45 minutes of skilled attention per
camera.**

### 2.2 The staff classifier is welded to this uniform

The apron/badge detector is not a general "is this person staff" model. It is:

- dark **navy** bib apron (`dark_v_max: 95`, `dark_s_max: 120`)
- a small circular **light** badge (`badge_v_min: 140` — chosen because this
  scene's 99th-percentile torso brightness is only ~156)
- shape gates measured off real crops from *this* clip: 0.5–6% of torso area,
  aspect ≥ 0.60, fill ≥ 0.42

A different chain, a different apron colour, a polo shirt instead of an apron,
or simply a lighting retrofit that lifts the scene's brightness distribution
breaks it. `badge_v_min` in particular is an **absolute** brightness threshold,
which is the least portable kind of number in the whole project.

It is also already the weakest link *within* this store — dropping its
threshold from 0.62 to 0.55 more than triples the reported session count.
Exporting a fragile component to 500 sites multiplies the fragility.

### 2.3 Thresholds tuned on 15 minutes of one store, with no ground truth

`task1.interest.score_threshold: 1.20` was chosen from worked examples of what
behaviour *should* qualify — not fitted to labels, because there are none. The
measured sensitivity is real:

| interest threshold | Total Interested |
|---:|---:|
| 1.10 | 12 |
| **1.20** | **6** |
| 1.30 | 4 |

That judgement was defensible for this store. Whether it holds at a busier
concourse, a narrower corridor, or a camera mounted 2 m higher is unknown, and
honesty requires saying so rather than assuming the number is universal.

Conversion (`Interested Entered`) is notably more robust — stable across every
entry rule tested — which suggests geometric events travel better than
attention-based ones. That is a useful prior for a rollout: trust the crossing
metrics first, calibrate the attention metrics per site.

### 2.4 The behavioural definitions may not be portable at all

This is the deepest issue and the least technical. "Interest" here is *facing +
slowing + approaching, near an open mall frontage with no door*. That
definition does not obviously mean the same thing at:

- a high-street store with a physical door and a display window
- a supermarket aisle, where everyone is close to a shelf and facing it
- an airport concourse, where nearly all traffic is transiting at speed

The **architecture** transfers to all three. The **definitions** need
re-deriving with the retailer, per store format. A rollout that copies the
thresholds without redoing that conversation will produce numbers that are
internally consistent and commercially meaningless.

---

## 3. Compute and storage reality

Measured on this project: 2 CPU cores processed 13,421 source frames (447 s of
video) in ~50 minutes of inference.

| | measured / derived |
|---|---|
| Analysis throughput | ~4.5 source-fps on 2 cores at `frame_stride: 3`, `imgsz: 960` |
| Ratio to real time | **~6.7× slower than real time** |
| CPU for one real-time camera | **~13 cores**, at these settings |
| GPU (T4-class) | roughly **5–8 cameras** per GPU at 10 Hz analysis; video *decode* usually becomes the bottleneck before inference does, so NVDEC matters |
| 1,500-stream fleet | ~250 GPUs centrally, **or** a Jetson-class edge box per 1–2 cameras |

Cheaper profiles are already available without touching code —
`yolo11n-pose.pt`, `imgsz: 768`, `frame_stride: 5` — at a recall cost on the
small figures in the walkway that would need measuring per site.

### Storage

| artefact | per camera-hour | note |
|---|---|---|
| Annotated H.264 video | ~415 MB | **debug artefact — do not ship this in production** |
| Feature cache (JSONL) | ~58 MB | worth a 7-day rolling buffer at the edge; enables re-tuning and dispute resolution |
| Event log (JSONL) | kilobytes | the only thing that needs to leave the store |

Writing annotated video for 1,500 cameras would be ~15 TB/day and is pure
waste. In production the pipeline emits **events**, and video is generated on
demand for a named incident — which the current code already supports, since
the annotation pass is separate from the inference pass and reads the cache.

---

## 4. What it would take to be fleet-ready

Ordered by leverage. Items 1 and 2 are the whole game: together they turn
*"a person configures each camera"* into *"a person confirms each camera"*.

### 1. Kill the hand-drawn zones — *~2–3 weeks*

Two levels, and the second is the real answer:

**Semi-automatic proposal.** Run detection unattended for an hour, cluster the
foot positions, and auto-propose the walkway/store split from the traffic dead
band. This is exactly what I did by hand during calibration, and it is
mechanisable — the dead band between walkway traffic and in-store traffic is a
density minimum, not a judgement call. A human then confirms or nudges in a UI.
**30–45 minutes per camera becomes ~5.**

**Store-plan coordinates.** Better still: a per-camera homography from four
clicked points on known fixtures, with zones defined **once in store-plan
coordinates** and projected into every camera that sees them. Zones become a
chain-level asset attached to the floor plan, not a camera-level chore. A
camera that is moved is re-homographed in two minutes and every zone follows.
This also gives genuine metric distances, which would let thresholds be stated
in metres instead of body-heights.

### 2. Replace the apron heuristic with a learned classifier — *~1 week + labelling*

A few hundred labelled torso crops training a linear probe on frozen CLIP or
DINOv2 features. A new uniform then costs an afternoon of labelling rather than
a rewrite, and it removes the absolute-brightness dependency that makes the
current version lighting-fragile. This is already the top item in the README's
"what I would do next", for the same reason.

### 3. Fit thresholds per store *archetype* — *~1 week + labelling per archetype*

Not per store (unaffordable) and not globally (wrong). Label 20–50 events for
each archetype — mall corridor, high-street frontage, supermarket aisle — and
fit the attention thresholds against those. The feature cache makes the sweep
itself free; the only missing ingredient is labels.

### 4. Drift monitoring — *~3–5 days*

Per camera, track detections/hour, mean person height, and the fraction of
detections falling inside each zone. A step change means the camera moved, the
lighting changed, or a fixture was reset, and the zones are now stale. **Without
this, a fleet degrades invisibly** — every camera keeps emitting confident,
wrong numbers.

### 5. Events, not video, as the default output — *~2 days*

Emit the event JSONL; generate annotated video only on request for a specific
window. The two-pass architecture already separates these, so this is a
configuration and retention-policy change rather than a rewrite.

---

## 5. Per-camera onboarding, before and after

| step | today | after items 1–2 |
|---|---|---|
| Locate the camera on the store plan | — | 2 min (click 4 fixture points) |
| Dump grid frame, run traffic scan | 10 min | automatic, unattended |
| Draw 3–7 polygons | 15 min | zones projected from the plan |
| Render preview and verify | 5 min | 3 min confirm/nudge in a UI |
| Smoke run and sanity check | 10 min | automatic, flagged only if anomalous |
| Uniform setup | rewrite thresholds | already covered by the chain's classifier |
| **Total skilled attention** | **30–45 min** | **~5 min** |

At 1,500 cameras that is the difference between roughly **1,000 person-hours**
and **125** — and, more importantly, between a task that needs someone who
understands the pipeline and one that needs someone who can look at a picture.

---

## 6. What stays hard

- **The definitions are a business conversation, not an engineering one.** What
  counts as "interest" at a given store format is the retailer's call. The
  pipeline can measure any definition consistently; it cannot choose the right
  one.
- **Small counts stay noisy.** This store yielded single-digit numbers over
  7.5 minutes. Fleet-level aggregates will be stable; a single store on a
  single afternoon will not be, and any dashboard must present confidence
  accordingly rather than implying precision it does not have.
- **Privacy and legal vary by jurisdiction.** Local-only processing helps a
  great deal, but retention of the feature cache — which contains keypoints,
  and is therefore biometric-adjacent in some regimes — needs a considered
  policy, not a default.
- **Ground truth does not scale linearly.** Labelling is the bottleneck on
  every accuracy claim in this document, and it is the one thing that cannot be
  automated away.

---

## Summary for an interview

> *Scene configuration and the uniform classifier are the two things standing
> between this and a fleet deployment. Everything above them — the body-height
> normalisation, the shared state machine, the config layering, the feature
> cache — was built assuming more than one store existed. What I would not
> claim is that the thresholds transfer: they were chosen defensibly for this
> store, against no ground truth, and the sensitivity tables in the README show
> exactly how much they matter.*
