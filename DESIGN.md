---
name: Distracted Driver Detection, live demo
description: A live, left-to-right architecture figure of MobileNetV3 on a projector-dark graphite ground, where real activations fill each slab as the signal passes.
colors:
  ground: "#0f1011"
  panel: "#16171a"
  panel-2: "#1d1f23"
  slab-bed: "#141518"
  rack-ground: "#121315"
  rule: "#2b2d31"
  ink: "#eef2f6"
  ink-2: "#b4bfcc"
  ink-3: "#8592a3"
  signal: "#7fe3cf"
  signal-dim: "#1f6f66"
  random: "#c97a10"
  driver: "#3d8fe0"
  ok: "#4cc38a"
  bad: "#f06a5f"
typography:
  display:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "2.3rem"
    fontWeight: 760
    lineHeight: 1.1
    letterSpacing: "-0.01em"
    fontVariation: "'wdth' 108"
  headline:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "1.55rem"
    fontWeight: 760
    lineHeight: 1.1
    letterSpacing: "-0.012em"
    fontVariation: "'wdth' 112"
  lede:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "1.3rem"
    fontWeight: 700
    lineHeight: 1.25
    fontVariation: "'wdth' 106"
  title:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "1.1rem"
    fontWeight: 720
    lineHeight: 1.2
  stage-name:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "0.95rem"
    fontWeight: 680
    lineHeight: 1.1
    fontVariation: "'wdth' 92"
  body:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "1.05rem"
    fontWeight: 400
    lineHeight: 1.35
  body-secondary:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "0.9rem"
    fontWeight: 400
    lineHeight: 1.35
  data:
    fontFamily: "JetBrains Mono, ui-monospace, Menlo, monospace"
    fontSize: "0.84rem"
    fontWeight: 400
    lineHeight: 1.35
    fontFeature: "'tnum' 1"
  data-figure:
    fontFamily: "JetBrains Mono, ui-monospace, Menlo, monospace"
    fontSize: "1.2rem"
    fontWeight: 600
    lineHeight: 1.1
    fontFeature: "'tnum' 1"
rounded:
  slab: "2px"
  control: "0.4rem"
  panel: "0.6rem"
  rail-end: "0 4px 4px 0"
  pulse: "50%"
spacing:
  hair: "0.25rem"
  xs: "0.5rem"
  sm: "0.8rem"
  md: "1.2rem"
  gutter: "1.5rem"
components:
  tab:
    textColor: "{colors.ink-2}"
    typography: "{typography.body-secondary}"
    rounded: "{rounded.control}"
    padding: "0.55rem 0.9rem"
  tab-active:
    textColor: "{colors.ink}"
  model-switch:
    backgroundColor: "{colors.panel}"
    rounded: "0.5rem"
    padding: "0.2rem"
  model-switch-option:
    textColor: "{colors.ink-2}"
    rounded: "0.35rem"
    padding: "0.45rem 0.8rem"
  model-switch-option-checked:
    backgroundColor: "{colors.panel-2}"
    textColor: "{colors.ink}"
  button-icon:
    backgroundColor: "{colors.panel-2}"
    textColor: "{colors.ink}"
    rounded: "0.45rem"
    size: "2.4rem"
  button-primary:
    backgroundColor: "{colors.signal}"
    textColor: "{colors.ground}"
    rounded: "{rounded.control}"
    padding: "0.55rem 1rem"
  button-ghost:
    textColor: "{colors.ink-2}"
    rounded: "{rounded.control}"
    padding: "0.55rem 1rem"
  rack-tile:
    textColor: "{colors.ink-2}"
    rounded: "0.45rem"
    padding: "0.3rem 0.6rem"
    width: "9.8rem"
  rack-thumb:
    rounded: "0.35rem"
    width: "4.8rem"
  panel-card:
    backgroundColor: "{colors.panel}"
    rounded: "{rounded.panel}"
    padding: "1rem"
  output-gauge:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink-2}"
    rounded: "{rounded.rail-end}"
    height: "1.25rem"
  output-gauge-winner:
    backgroundColor: "{colors.panel-2}"
    textColor: "{colors.ink}"
  verdict-answer:
    textColor: "{colors.ink}"
    typography: "{typography.display}"
  confusion-cell:
    textColor: "{colors.ink-2}"
    typography: "{typography.data}"
    rounded: "{rounded.slab}"
---

# Design System: Distracted Driver Detection, live demo

## Overview

**Creative North Star: "The Page Is the Network"**

The system is a CNN architecture figure in the LeNet/AlexNet slab-diagram tradition, made live. One left-to-right axis carries the image from input photo through six stacked feature-map slabs, a pooled strip and ten output gauges to a locked verdict. Tensors are drawn as decks of real activation maps seen at an angle, labelled with their dimensions in tabular mono, and a mint pulse travels the wire that joins them. Everything sits on a projector-dark graphite ground so that colour is reserved for meaning: mint for the signal, amber and blue for the two models, green and red only for correct and wrong.

Density is that of a technical figure, not a dashboard. Numbered stages (01 to 09), plain captions under each slab and one narration line beneath the figure explain what is on screen, so a visitor can read it without a presenter. The type scales with the viewport so the same layout reads from the back of a classroom at 1920 wide. Depth is used only where the figure needs it: perspective on the slab decks and a lifted input photo. Panels are flat, ruled and tonal.

The world explicitly refuses the card dashboard with a hero bar chart and any metaphor world; the Results view uses panels for evidence, but the first view is the figure itself.

**Key Characteristics:**
- One left-to-right axis: input, six slab decks, pooled strip, ten gauges, verdict.
- Real activations and Grad-CAM drawn pixel-honest on a single mint signal ramp.
- Graphite ground with two tonal panel steps and a single hairline rule colour.
- Archivo with its width axis for every word; JetBrains Mono for every measured number.
- Fluid root size so the whole figure scales for the projector.
- Model identity colours always travel with their name.

## Colors

A neutral graphite world where hue is functional: one signal hue, one identity pair, one status pair.

### Primary
- **Signal Mint** (signal): the activation travelling through the network. The pulse dot, the current step number, the active tab underline, focus rings, selection, slider and checkbox accents, the selected rack photo border, hover on icon buttons and tiles, and the highlighted words in the comparison lede. The primary button fill.
- **Deep Teal Trace** (signal-dim): the lit portion of the wire behind the pulse; the wire that has already been travelled.

The activation and Grad-CAM ramp is one hue: ground to deep teal to Signal Mint to white-hot (stops at 0, 0.3, 0.6, 0.85, 1; defined in the sidecar). Grad-CAM darkens the photo where the model did not look and paints this ramp where it did.

### Secondary (model identity pair)
- **Random-Split Amber** (random): the random-split model, everywhere it appears: switch swatch, reader heading swatch, top-3 tracks, chart series, its confusion matrix shading.
- **Driver-Split Blue** (driver): the driver-split model, with the same reach. The output gauges inside the network take whichever model is shown.

The pair was validated as colour-blind safe on the panel colour.

### Tertiary (status)
- **Correct Green** (ok): the check icon on a correct verdict.
- **Wrong Red** (bad): the cross icon on a wrong verdict.

### Neutral
- **Projector Graphite** (ground): the page.
- **Panel** (panel): narration bar, drawer, readers, chart cards, model switch track, gauge rails, the input photo bed.
- **Panel Raised** (panel-2): checked switch option, icon buttons, winning gauge row, highlighted table row, top-3 track rail.
- **Slab Bed** (slab-bed): behind feature-map canvases and the pooled strip before they light.
- **Rack Ground** (rack-ground): the photo rack band along the bottom, one step below the ground's neighbours.
- **Hairline** (rule): every 1px border, the header and rack dividers, table rules, the unlit wire.
- **Ink** (ink), **Ink 2** (ink-2), **Ink 3** (ink-3): primary text and lit stage names; secondary text, captions and unlit stage names; tertiary metadata, step numbers, dimension labels and axis headers.

### Named Rules
**The Mint Is Only the Signal Rule.** Signal Mint and its ramp mean "activation, attention, or where you are". Never use amber or blue for activations or heat; those hues belong to the models.

**The Named Identity Rule.** Random-Split Amber and Driver-Split Blue never appear without the model's name beside them (switch label, reader heading, legend, chart card title). A swatch alone is not a label.

**The Icon Plus Word Rule.** Correct and wrong are always an icon and a word ("Correct" / "Wrong") with the true label spelled out; green and red are never the only carrier.

## Typography

**Display Font:** Archivo variable (with Helvetica Neue, Arial), self-hosted, weight 100 to 900, width 62% to 125%
**Label/Mono Font:** JetBrains Mono variable (with ui-monospace, Menlo), self-hosted

**Character:** Archivo's width axis does the hierarchy work: wide and heavy (106 to 112%) for the title, verdict and headline statements, condensed (92%) for the stage names that must fit under narrow slabs. JetBrains Mono with tabular figures carries every measurement: dimensions, step counters, percentages, class codes, CPU milliseconds, table cells.

### Hierarchy
- **Display** (760, 2.3rem, 1.1, wide): the locked verdict only.
- **Headline** (760, 1.55rem, 1.1, widest): the page title; the Results headline sits at the same weight (1.5rem).
- **Lede** (700, 1.3rem, 1.25, slightly wide): the one-sentence claim at the top of Two models, max 70ch.
- **Title** (700 to 720, 1rem to 1.15rem): reader headings, chart card titles, drawer title, rack heading.
- **Stage name** (680, 0.95rem, condensed): numbered node names in the figure; Ink 2 until lit, Ink when lit.
- **Body** (400, 1.05rem): the narration line, max 62ch. **Body secondary** (0.9rem, Ink 2): captions, notes, byline, model note.
- **Data** (mono, 0.84rem, tabular): step numbers, dimensions, class codes, table cells. **Data figure** (mono 600, 1.2rem): the confidence beside the verdict; the CPU readout uses 1.05rem bold.

### Named Rules
**The Every Number Is Mono Rule.** If it was measured or computed, set it in JetBrains Mono with tabular figures. Words stay in Archivo.

**The Projector Floor Rule.** The root size is fluid, clamp(14px, 1.25vw, 26px): 16px at 1280 wide, 24px at 1920. Every size is in rem so it rides that clamp. Readable text holds at 0.8rem and above; data labels at 0.84rem.

## Layout

The Inside view is a single horizontal figure: input node (10.4rem), six stage columns whose deck size shrinks with the real tensor (5.8rem down to 3.8rem), a pooled node (4.2rem), and an output column (15.5rem), distributed with space-between. All nodes share a fixed header height (2.9rem) and visual height (9.6rem) so the wire runs through every vis centre on one line. Below the figure, a narration bar holds step controls, the narration line and the right-aligned verdict. The photo rack is pinned to the bottom of the viewport (margin-top auto in a column-flex main) and is shared by the first two views.

Two models is a two-column grid of readers (photo 15rem plus text). Results is a 1.25fr / 1fr grid of chart cards followed by a two-up row of confusion matrices.

Page gutter is 1.5rem; gaps step through 0.5, 0.8, 1.2 and 1.5rem. Below 900px the root locks to 15px, columns collapse to one, tabs wrap to their own row, and the figure and narration keep a 70rem minimum width and scroll sideways instead of squashing.

**The One Axis Rule.** The network reads left to right on one wire. Never stack the stages vertically or break the figure into cards; on narrow screens it scrolls.

## Elevation & Depth

Flat and tonal by default: ground, Panel and Panel Raised separated by hairline rules. Depth appears only to serve the figure. The slab decks sit in perspective (46rem, rotateY -30deg, rotateX 6deg), each map pushed back along Z, and straighten slightly on hover. The current deck's front map gets a soft mint glow. The input photo is lifted off the ground. Floating layers (tooltip) cast one soft shadow.

### Shadow Vocabulary
- **Photo lift** (`box-shadow: 0 .4rem 1.4rem rgba(0,0,0,.55)`): the input photo canvas.
- **Current slab glow** (`filter: drop-shadow(0 .3rem .9rem rgba(127,227,207,.45))`): front map of the current stage only.
- **Switch thumb** (`box-shadow: 0 1px 3px rgba(0,0,0,.5)`): the checked option in the model switch.
- **Float** (`box-shadow: 0 .3rem 1rem rgba(0,0,0,.5)`): chart tooltip.

**The Depth Belongs to Tensors Rule.** Perspective and glow are for the data objects in the figure. Panels, buttons and charts stay flat.

## Shapes

Small, precise corners. Feature maps, swatches and confusion cells are nearly square (2px), echoing pixel grids. Controls round at 0.35 to 0.45rem, panels at 0.6rem. Gauges and chart bars start square on the baseline and round only at the far end (0 4px 4px 0), so length reads from a hard zero. The pulse is the only circle. Dashed borders mark drop targets: the rack tiles and the full-screen dropzone.

## Components

### Tabs
- Text-only, Ink 2 at 0.95rem 600; hover adds a Panel fill. Selected turns Ink with a 3px Signal Mint underline inset to the label. ARIA tablist semantics.

### Model switch
- A Panel track with a 0.2rem inset holding two radio options, each a 0.8rem square identity swatch plus the full model name. Checked option is Panel Raised with Ink text and the thumb shadow. The shown model sets the gauge colour.

### Slab deck (signature)
- Eight real feature maps per stage, stacked in perspective with the stage's `--deck` size. Unlit: 14% opacity, desaturated and dimmed on the Slab Bed. When the pulse reaches the stage, maps fade in with a 45ms stagger per map, each layer slightly less opaque than the one in front. Click or Enter opens the drawer of all eight maps at true resolution.

### Wire and pulse
- A 2px hairline wire through every vis centre; the travelled part turns Deep Teal Trace and a 0.75rem Signal Mint dot (ringed in ground) moves to the current node. Both ride the standard ease over 0.5s.

### Output gauges
- Ten rows: mono class code, a rail with the class name on it, mono percentage. The fill is the shown model's colour at 38% opacity; the winner row lifts to Panel Raised, bold Ink, fill at 62%. The true class carries a small "true" tag.
- Gauges settle with a damped spring (stiffness 170, damping 15) that may overshoot slightly; the verdict locks at 700ms or when the spring rests, and final percentages are written then.

### Verdict
- Right-aligned in the narration bar: class name in Display, mono confidence beside it, and a truth line below with the check or cross icon, the model name, "Correct" or "Wrong", and the true label. Uploaded and webcam photos say they have no true label.

### Narration bar and icon buttons
- Panel bar with previous, replay and next as 2.4rem square icon buttons (Panel Raised, hairline border, mint on hover), then a mono "NN / 09" step counter in mint before the narration sentence.

### Photo rack
- Horizontally scrolling 4:3 thumbnails (4.8rem) with a mono class-code tag, slightly dimmed until hover or selection; the selected photo gets a 2px mint border. The strip fades out at its right edge; scroll arrows sit in the rack header. Upload and webcam are dashed tiles (icon plus label) that turn mint-bordered on hover.

### Buttons
- **Primary:** Signal Mint fill, ground text, 700. **Ghost:** transparent with hairline border, Ink 2. Both 0.55rem by 1rem, 0.4rem corners. Used in the webcam panel.

### Reader (Two models)
- Panel card per model: Grad-CAM photo, heading with identity swatch and model name, training note, answer and truth line, top-3 with rails in the model colour, and mono facts.

### Charts and table (Results)
- Panel cards with a title, an Ink 3 note and a legend that pairs each swatch with the split's full description. Grouped bars in the identity pair with mono value labels and a faint grid; hovering a bar shows an exact tooltip. The metrics table is right-aligned mono with the MobileNetV3 row raised.

### Confusion grid (Results)
- Native grid, not an image: row labels (code plus name), mono column codes, square-ish cells. Each cell is shaded in its model's colour by its share of the true-class row (8% floor to full), with ground-coloured numbers on dark-enough shades and Ink 2 otherwise. Each matrix's card title names the split.

### Motion
- Standard ease cubic-bezier(.16, 1, .3, 1). Autoplay steps every 820ms through nine steps; Space replays, arrow keys step. Under prefers-reduced-motion all CSS transitions and delays drop to zero. The `?still` query adds a class that disables transitions for screenshot capture.

## Do's and Don'ts

### Do:
- **Do** keep activations and Grad-CAM on the single mint ramp, drawn at true resolution with pixelated scaling.
- **Do** pair every Random-Split Amber or Driver-Split Blue mark with the model's name.
- **Do** show correct and wrong as an icon plus the word, with the true label written out.
- **Do** read every number from metrics.json or a live prediction; never type one into the page.
- **Do** set measurements in JetBrains Mono with tabular figures and size everything in rem under the fluid root.
- **Do** keep the figure on one left-to-right axis and let it scroll sideways on narrow screens.

### Don't:
- **Don't** use amber or blue for activations, heatmaps or the signal; those hues are the models.
- **Don't** tell correct from wrong, or the two models apart, by colour alone.
- **Don't** reduce the network view to a card dashboard with a bar chart.
- **Don't** add perspective, glow or shadow to panels and controls; depth belongs to the tensors.
- **Don't** set readable text below 0.8rem.
