# Calibration evidence

These are the artefacts behind the claims in the main README §3 ("How the
zones were calibrated") and §7 ("Identifying staff"). They are working
material, not deliverables — but they are what makes the configuration
defensible rather than guessed.

| file | what it shows |
|---|---|
| `calibration_entrance_traffic.jpg` | Every foot position detected across the **whole** `entrance.mp4` clip at 2 Hz (3,017 detections), plotted onto a reference frame. Colour encodes person height: red < 80 px, amber 80–160 px, green > 160 px. Two things are visible: essentially no detections above y ≈ 225 (so the ROI cut is safe), and a clear dead band between walkway traffic and in-store traffic where the store's front-line fixtures stand. The storefront line is drawn through that dead band. |
| `calibration_interior_traffic.jpg` | The same for `interior.mp4` (1,357 detections). Shows the dense service-counter cluster at y ≈ 130–230 that the ROI excludes, and the aisles where genuine shelf engagement happens. |
| `zones_entrance.jpg` | The configured `store_polygon`, `walkway_polygon` and `storefront_line` rendered onto a real frame. Reproduce with `python -m rva.cli zones --config configs/entrance.yaml --video data/entrance.mp4`. |
| `zones_interior.jpg` | The configured shelf A–D polygons and the ROI cut. Reproduce with `python -m rva.cli zones --config configs/interior.yaml --video data/interior.mp4 --at 25`. |
| `staff_apron_badge_closeup.png` | A 5× nearest-neighbour blow-up of a staff torso crop at t = 10 s. The dark navy bib apron and the circular light badge with its dark logo are both clearly visible; this is the crop every badge parameter was measured from (20 × 17 px blob, 3.6 % of the torso, aspect 0.85, fill 0.50). |
| `staff_badge_detector_check.jpg` | The badge detector run over torso crops sampled across the whole clip. Green border = badge detected, red = not. The staff member's crops fire; customer crops (including several in dark clothing) do not. |

Regenerating the traffic plots is a two-step job and is not part of the
pipeline, since it is a one-off scene-setup activity: run the detector over the
clip collecting `(frame, x_centre, y_bottom, height)` per detection, then scatter
those points over a reference frame. The zone previews and coordinate grids
*are* part of the CLI (`rva.cli zones`, `rva.cli grid`) because they need to be
re-run whenever a config is edited.
