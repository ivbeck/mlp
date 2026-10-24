---
name: mlp console
description: A local training console that draws a perceptron's output field as a surface-pressure weather chart.
colors:
  chart-paper: "#e9eef2"
  panel: "#f4f7f9"
  sheet-white: "#fbfcfd"
  ink: "#1d2733"
  ink-soft: "#4a5663"
  grid-grey: "#8a96a3"
  hairline: "#c9d1d9"
  warm-front: "#d2362b"
  warm-ink: "#b02a20"
  cold-front: "#2459a8"
  steady-green: "#2f7d4a"
typography:
  display:
    fontFamily: "Fira Sans, system-ui, sans-serif"
    fontSize: "56px"
    fontWeight: 500
    lineHeight: 1
    letterSpacing: "-0.02em"
    fontFeature: "tnum"
  headline:
    fontFamily: "Fira Sans, system-ui, sans-serif"
    fontSize: "28px"
    fontWeight: 500
    lineHeight: 1.15
    letterSpacing: "-0.01em"
    fontFeature: "tnum"
  title:
    fontFamily: "Fira Sans, system-ui, sans-serif"
    fontSize: "20px"
    fontWeight: 500
    fontFeature: "tnum"
  body:
    fontFamily: "Fira Sans, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.45
    fontFeature: "tnum"
  body-small:
    fontFamily: "Fira Sans, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.45
    fontFeature: "tnum"
  label:
    fontFamily: "Fira Sans Condensed, Fira Sans, system-ui, sans-serif"
    fontSize: "11.5px"
    fontWeight: 500
    letterSpacing: "0.08em"
  chart-numeral:
    fontFamily: "Fira Sans, system-ui, sans-serif"
    fontSize: "11px"
    fontWeight: 500
    fontFeature: "tnum"
rounded:
  flag: "2px"
  r: "3px"
spacing:
  hair: "4px"
  tight: "8px"
  row: "10px"
  block: "14px"
  section: "22px"
  gut: "24px"
  gut-phone: "16px"
  side: "364px"
  header: "56px"
components:
  button:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.r}"
    padding: "0 16px"
    height: "38px"
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.chart-paper}"
    typography: "{typography.body}"
    rounded: "{rounded.r}"
    padding: "0 16px"
    height: "38px"
  button-primary-hover:
    backgroundColor: "#2c3a4a"
  button-danger:
    backgroundColor: "transparent"
    textColor: "{colors.warm-ink}"
    rounded: "{rounded.r}"
    height: "38px"
  input:
    backgroundColor: "{colors.sheet-white}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.r}"
    padding: "0 8px"
    height: "34px"
  file-chip:
    textColor: "{colors.ink}"
    typography: "{typography.body-small}"
    rounded: "{rounded.r}"
    padding: "3px 10px"
  flag:
    textColor: "{colors.warm-ink}"
    typography: "{typography.label}"
    rounded: "{rounded.flag}"
    padding: "2px 6px"
  segmented-on:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.chart-paper}"
    height: "26px"
    padding: "0 10px"
  header:
    backgroundColor: "{colors.chart-paper}"
    textColor: "{colors.ink}"
    height: "{spacing.header}"
    padding: "0 24px"
  side-panel:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink}"
    width: "{spacing.side}"
    padding: "20px 24px 28px"
  run-strip:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink}"
    padding: "8px 24px"
  isoline-label:
    backgroundColor: "{colors.chart-paper}"
    textColor: "{colors.ink}"
    typography: "{typography.chart-numeral}"
    height: "18px"
  front-label:
    backgroundColor: "{colors.warm-front}"
    textColor: "#ffffff"
    typography: "{typography.chart-numeral}"
    height: "18px"
---

# Design System: mlp console

## Overview

**Creative North Star: "The Surface Analysis"**

The console is a weather-service surface-pressure chart that happens to show a neural network. The model's output over two features is drawn as a field: black isolines with boxed numerals, the p = 0.5 decision boundary as a stationary front (red line with semicircles, blue dashes with triangles), data rows as station circles, and the field's high and low marked like pressure centres. Loss over time is a barograph trace. Everything around the chart is printed paper: a pale blue-grey ground, dark ink, ruled frames, condensed caps for labels.

Density is that of a working instrument, not a dashboard. One large chart owns the left of the screen, a 364px ruled side panel holds the counter, the numbers and the controls, and nothing floats. Depth comes from rules and tone, never from shadow. The only motion is the field easing towards each new snapshot so the front moves like a weather loop, the scrubber, and small value changes.

The world refuses the experiment-tracker look: no grid of metric tiles, no line-chart cards, no cards inside cards.

**Key Characteristics:**
- Chart-paper ground, ink rules, two front colours that carry meaning (warm = class 1 / high, cold = class 0 / low).
- Fira Sans with tabular figures everywhere; Fira Sans Condensed uppercase for small labels and column heads.
- 1.5px ink rules frame regions; 1px hairlines divide rows; dotted grids sit inside frames.
- Square-ish 3px corners; flat surfaces; no shadows.
- Numbers on lines are boxed, the front's numeral is reversed out of red.

## Colors

A cool, low-chroma paper and ink system with exactly two saturated voices, the warm and cold fronts, which always encode data or state.

### Primary
- **Weather-Map Ink** (ink): text, 1.5px region rules, chart frames, isolines, class-1 station fill, primary button fill, active tab underline, selection background.

### Secondary
- **Warm-Front Red** (warm-front): the warm half of the stationary front and its semicircles, the reversed front label, validation-loss trace, the high side of the field tint, danger button border, error state, sample-data tag border.
- **Warm Ink** (warm-ink): red used as text, where warm-front would be too light (validation loss numeral, warnings, failed state, danger button label, error banner text).

### Tertiary
- **Cold-Front Blue** (cold-front): the cold half of the front (dashed 42/42) and its triangles, the low side of the field tint, negative weights, paused / stopped-early state, the 2px focus ring, inline text links.
- **Steady Green** (steady-green): only the live stream dot and the "Training" state. Not a general success colour.

### Neutral
- **Chart Paper** (chart-paper): page ground and the canvas of every chart; also the fill of class-0 stations and isoline label boxes.
- **Panel Paper** (panel): side panel, run strip, info banner.
- **Sheet White** (sheet-white): input fields and the selected row in lists and tables.
- **Soft Ink** (ink-soft): secondary text, labels, tick numerals, meta lines.
- **Grid Grey** (grid-grey): dotted chart grids, control borders (inputs, chips, segmented controls), dashed drop zones, comparison traces of other runs.
- **Hairline** (hairline): 1px row dividers, the fine grid of the barograph, the empty progress bar.

### Named Rules
**The Two Fronts Rule.** Warm and cold are reserved for meaning: class 1 / high values / validation / danger on the warm side, class 0 / low values / negative / paused on the cold side. Never use either as decoration or brand colour.

**The Data-Ramp Rule.** Colour ramps appear only where they encode a value: the field tint (paper mixed at most 11% towards warm or cold), the regression station fill and its key ramp, the probe meter. No decorative gradients anywhere else.

## Typography

**Display Font:** Fira Sans 500/600 (self-hosted, fallback system-ui)
**Body Font:** Fira Sans 400/500 (fallback system-ui)
**Label Font:** Fira Sans Condensed 500, uppercase

**Character:** A technical humanist sans with tabular figures throughout, so numbers in counters, tables and tick marks line up like a printed chart. The condensed caps play the role of the small legends printed on a map.

### Hierarchy
- **Display** (500, 56px, line-height 1, -0.02em): the epoch counter in the side panel, with the total in a 24px grey `small`; 52px for the probe verdict, 44px on phones.
- **Headline** (500, 28px, 1.15): view titles ("Surface analysis · epoch 38"), the varying part set in 600. 22px on phones. Loss numerals in the side panel use the same 28px size.
- **Title** (500, 20px): titles of secondary charts (Barograph, activations).
- **Body** (400, 14px, 1.45): default text, buttons (500), inputs. Notes cap at 62ch.
- **Body small** (400, 13px / 12.5px): meta lines, file chip, key items, run descriptions.
- **Label** (Condensed 500, 11 to 12px, uppercase, 0.06 to 0.08em): side-panel section labels, table heads, run state, layer numbers, axis caps under the network diagram, flags.
- **Chart numeral** (500, 11px): boxed isoline values; tick numerals 11px 400 in soft ink; axis names Condensed 500 12px, 0.04em.

### Named Rules
**The Tabular Rule.** `font-variant-numeric: tabular-nums` is set on the body and never turned off. Every numeral in this world is a reading.

## Layout

A two-column shell under a 56px sticky header: the work area (`minmax(0, 1fr)`, padding 18px 24px 28px, 14px vertical gap) and a fixed 364px side panel, separated by a 1.5px ink rule and filling `100dvh - 56px`, each scrolling on its own. The side panel stacks sections with a 22px gap, 10px inside a section; its primary action sits at the bottom (`margin-top: auto`) on setup views and at the top on the Train view.

On Train the surface chart is 470px tall and the barograph 180px below it. Charts use fixed inner margins (48px left for tick numerals, 14px right) and fill the column width.

Below 1080px the side panel drops under the work area with a 1.5px rule on top, and a sticky run strip (panel paper, ink rule below) appears at the top of the work area carrying the epoch counter and the Pause / Step / Stop buttons; the side panel's own counter and controls hide. Below 640px the gutter shrinks to 16px, the header wraps with the tabs on their own 44px scrolling row, plane selects go full width and the chart readout becomes a static line under the chart.

Spacing is small-step and dense: 4, 8, 10, 14, 22, 24px. Forms are a two-column grid with 10px x 12px gaps.

## Elevation & Depth

Flat. There are no shadows anywhere. Depth is tonal and ruled: the side panel and run strip sit on a slightly lighter paper (panel) behind a 1.5px ink rule; selected rows lift to sheet white with a 1px ink outline. The only stacking is the sticky header and run strip, which keep their opaque paper and rule instead of a shadow.

### Named Rules
**The Ruled-Not-Raised Rule.** Separate regions with a 1.5px ink rule, rows with a 1px hairline, and selection with sheet white plus a 1px ink outline. Never with a shadow, a card or a nested box.

## Shapes

Near-square corners: 3px on buttons, inputs, chips, segmented controls and dashed empty/drop zones, 2px on flags and tags, 0 on charts, tables, list rows and the run strip. Chart frames are sharp 1.5px ink rectangles. Circles are reserved for meaning: stations (r 4.2, 1.5px ink stroke), the best-epoch ring on the barograph, the probe crosshair ring, the stream dot. Dashed outlines mean "something can go here" (drop zone, empty state, Add hidden layer button).

## Components

### Buttons
- **Shape:** gently squared (3px), 38px tall, 16px side padding, 1.5px ink outline, Fira Sans 500 14px.
- **Default:** transparent with ink text; hover tints with 7% ink.
- **Primary:** ink fill, paper text; hover deepens to a lighter slate (#2c3a4a). The main action in the side panel is 44px tall at 15px.
- **Danger:** warm-front border, warm-ink text, 8% warm hover. Used for Stop.
- **Link:** cold-front text, underlined (offset 3px), 13px 500, no box.
- **Icon:** 30px square, 1px grid-grey border, 16px stroked SVG (1.6 stroke).
- **Disabled:** 45% opacity.
- **Focus:** 2px cold-front outline, 2px offset, on every focusable element.
- **Transitions:** background and colour, 120ms.

### Chips and flags
- **File chip:** 1px grid-grey border, 3px radius, 13px text, truncates at 46ch. Holds file, row count, feature count and target.
- **Flag / tag:** Condensed uppercase 11px, 1px border, 2px radius; warm border and warm-ink text for sample data and warnings, grid grey otherwise.
- **Stream status:** 8px dot plus soft-ink text; green while streaming, warm when reconnecting, grey otherwise.

### Inputs / Fields
- **Style:** 34px, sheet-white fill, 1px grid-grey border, 3px radius, 8px padding. Field label above in 12.5px soft ink.
- **Hover:** border goes ink. **Focus:** the global 2px cold ring. **Disabled:** paper fill, soft-ink text.
- **Segmented control:** 1px grid-grey border, 26px buttons at 12.5px; the selected segment is filled ink with paper text.
- Checkboxes and radios are native at 16px, tinted ink by `accent-color`.

### Navigation
- Header tabs (Data, Network, Train, Probe) fill the 56px header height, 500 14px soft ink; the active tab goes ink with a 3px ink underline sitting on the header's 1.5px rule. Disabled steps stay readable in a mid grey, not faded out.

### Surface Chart (signature)
The feature-plane chart. A canvas tint underneath (paper eased at most 11% towards warm above the boundary and cold below), a dotted grid-grey grid (0.6px, 2/3 dash), soft-ink tick numerals, Condensed axis names with arrows at the far ends of the axes. Isolines are 1.05px ink, Chaikin-smoothed, spaced so no two run closer than about 18px, each carrying one boxed numeral (paper box, 1px ink border, 18px tall). The p = 0.5 isoline becomes the stationary front: a 2.6px warm line under a 2.6px cold line dashed 42/42, with cold triangles pointing into the 0 side and warm semicircles bulging into the 1 side, alternating every 42px; its numeral is reversed white out of red. High and low centres are 26px 600 numerals in warm-ink and cold with a 5px paper halo. Stations are 4.2px circles: hollow paper for 0, filled ink for 1, or ramp-filled for regression targets. The frame is a 1.5px ink rectangle. A readout box (paper, 1px ink border, 12.5px) sits top right and reads x, y and p under the pointer; on Probe a 2px ink ring-and-crosshair marks the probed point. Field changes tween over 160ms with an exponential ease-out; reduced motion swaps instantly.

### Barograph (signature)
The loss chart under the surface. A fine vertical hairline grid (0.5px), dotted horizontal grid, 1.5px ink frame. Train loss is a 1.5px ink trace, validation a 1.5px warm trace, other runs with the same loss a 1px grid-grey 3/3 dashed trace. The best epoch is a paper-filled warm ring (r 4.5). The scrubber is a 1px ink line with a solid ink triangle above the frame (1.6px while held), snapping to epochs with a stored field; dragging or arrow keys replay the field, End returns to live, and a "Back to live" link appears while scrubbed. Head row: 20px title, inline key items (22px line swatches) with live values, loss name and epoch meta, and a Linear / Log segmented control.

### Side Panel (signature)
Panel paper behind a 1.5px left ink rule, 364px wide. On Train: the 56px epoch counter over a 4px progress bar (hairline track, ink fill, 200ms width ease-out), the run state line, two loss blocks side by side each topped by a 1.5px ink rule (28px numerals, validation in warm-ink, accuracy under them), the run controls in a 1.4fr / 1fr / 1fr button grid, the network line, the chart key, the runs list and the next-run settings. Section labels are Condensed caps in soft ink.

### Run Strip (signature)
Below 1080px only. Sticky under the header (top 0 on phones), panel paper with a 1.5px ink rule below, running full bleed across the work area: the epoch as a 20px 600 numeral over the total, the run state in its state colour, then the same Pause / Step / Stop grid at 34px.

### Chart Key (signature)
A side-panel section explaining the marks with 24 x 14px inline SVG swatches drawn with the chart's own strokes: hollow and filled stations with row counts, a front swatch (warm half with semicircle, cold half with triangle), an isoline swatch, and for regression targets a 90 x 10px cold-paper-warm ramp bar with its min and max. 12.5px text, 7px gap.

### Runs List
Rows divided by hairlines, no boxes: run name 600, description and loss value on a 12.5px line, state in Condensed caps coloured by state. The selected run sits on sheet white with a 1px ink outline and a 2px underline on its name.

### Data Table
Full width under a 1.5px ink rule; Condensed uppercase heads over a 1px ink line, 13.5px cells over hairlines, numbers right aligned, the selected column row on sheet white with an ink outline.

## Do's and Don'ts

### Do:
- **Do** draw any new view of the model's output in the surface-chart vocabulary: ink isolines with boxed numerals, the front at the decision threshold, stations for rows.
- **Do** keep warm-front and cold-front for meaning (class 1 / class 0, high / low, positive / negative, validation / danger vs. paused / links).
- **Do** frame regions with the 1.5px ink rule and divide rows with 1px hairlines.
- **Do** set every numeral in Fira Sans with tabular figures, and small labels in Fira Sans Condensed caps.
- **Do** keep motion to the 160ms field tween, the scrubber, and 120 to 200ms value or state transitions, all off under reduced motion.
- **Do** show a key next to any chart whose marks are not self-evident, drawn with the chart's own strokes.

### Don't:
- **Don't** use shadows, cards or cards in cards; depth is rules and paper tone.
- **Don't** use gradients except as data ramps that encode a value.
- **Don't** build metric-tile grids or line-chart cards; numbers live in the side panel, trends on the barograph.
- **Don't** rely on colour alone on the chart: classes differ by fill (hollow vs. solid), front sides by symbol (semicircle vs. triangle), the cold front by its dash.
- **Don't** round corners beyond 3px.
