---
version: 1
slug: "web-ui-index-html"
primary_target: "web/ui/index.html"
related_targets: ["web/ui/src"]
---

# Surface: mlp console (web/ui)

Mode: Operate. Audience: people with their own numeric CSV training a small MLP locally.
Job: load data, shape the network, watch it train live, probe the result. Four views:
Data, Network, Train, Probe. Constraints: single output neuron (crate limit), numbers are
the crate's, light theme (desktop, daylight office, long watched sessions).

## Direction contract

THESIS: The model's output field is weather. The Train and Probe views draw p(y) over two
chosen features as a surface-pressure chart: labelled isolines, the 0.5 decision boundary as
a stationary front, data rows as station markers. Refuses the experiment-tracker grid of
metric tiles and line-chart cards.

OWN-WORLD: Chart-paper ground #e9eef2, ink #1d2733, warm-front red #d2362b, cold-front blue
#2459a8, grid grey #8a96a3, side panel #f4f7f9. Fira Sans (self-hosted) with tabular figures,
Fira Sans Condensed caps for small labels. 1.5px ink rules frame regions, dotted fine grids,
3px radii, boxed inline numerals on lines. No shadows, no gradients, no cards-in-cards.

STORY: The user drops a CSV, sees its columns and stations on the chart, picks a target,
sets layers, presses Train and watches fronts sweep across the plane while the barograph
falls; they scrub back through epochs, then click the map to probe any point and read
every layer's activations.

FIRST VIEWPORT: Header 56px (name, Data/Network/Train/Probe tabs, file chip, stream
status). Train: left ~75% the feature-plane chart (title "Surface analysis · epoch N",
x/y feature selects), under it the barograph (train ink, val red, scrubber). Right column
364px: epoch counter, train/val loss (+accuracy for 0/1 targets), Pause/Step/Stop, run
config, runs list. Primary action: Train/Pause in the right column top.

FORM: Surface analysis (weather-service surface-pressure chart), my top-ranked grounded
candidate (#1 of 7), taken as IMPECCABLE'S PICK. seed:184f515cf82a947d.
Signature interaction: isolines and front re-contoured every snapshot and tweened between
fields, so the front moves like a weather loop; dragging the barograph scrubber replays the
stored fields epoch by epoch. Motion grammar: field tweens 160ms ease-out, nothing else
animates except the scrubber and value changes.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
