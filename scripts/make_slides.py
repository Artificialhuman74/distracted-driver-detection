"""Build deliverables/slides.pptx from outputs/metrics.json and outputs/figures/*.png.

Every number on the slides is read from metrics.json; nothing is typed in by hand.
Usage:  .venv/bin/python scripts/make_slides.py [--outputs outputs] [--out deliverables/slides.pptx]
"""
import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--outputs", default=str(ROOT / "outputs"))
ap.add_argument("--out", default=str(ROOT / "deliverables" / "slides.pptx"))
args = ap.parse_args()
OUT = Path(args.outputs)
FIG = OUT / "figures"
M = json.loads((OUT / "metrics.json").read_text())

# ---------- design tokens (same world as the demo) ----------
GROUND = RGBColor(0x0B, 0x0E, 0x13)
PANEL = RGBColor(0x17, 0x1C, 0x25)
RULE = RGBColor(0x26, 0x30, 0x40)
INK = RGBColor(0xEE, 0xF2, 0xF6)
INK2 = RGBColor(0xB4, 0xBF, 0xCC)
INK3 = RGBColor(0x85, 0x92, 0xA3)
SIGNAL = RGBColor(0x7F, 0xE3, 0xCF)
RANDOM = RGBColor(0xC9, 0x7A, 0x10)
DRIVER = RGBColor(0x3D, 0x8F, 0xE0)
FONT = "Arial"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
W, H = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]
DEEP = "MobileNetV3"
pct = lambda v: f"{v * 100:.1f}%"
SMOKE = M["meta"]["smoke"]


def text(slide, x, y, w, h, s, size=20, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    lines = s if isinstance(s, list) else [s]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = line
        r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold; r.font.name = FONT
    return tb


def bullets(slide, x, y, w, h, items, size=20, gap=10):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True; tf.margin_left = 0
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        head, _, rest = it.partition("|")
        r = p.add_run(); r.text = head; r.font.size = Pt(size); r.font.name = FONT
        r.font.bold = bool(rest); r.font.color.rgb = INK
        if rest:
            r2 = p.add_run(); r2.text = " " + rest; r2.font.size = Pt(size); r2.font.name = FONT; r2.font.color.rgb = INK2
    return tb


def rect(slide, x, y, w, h, fill, line=None):
    s = slide.shapes.add_shape(1, x, y, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None: s.line.fill.background()
    else: s.line.color.rgb = line
    s.shadow.inherit = False
    return s


def new_slide(title, n, note=""):
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = GROUND
    text(s, Inches(0.7), Inches(0.45), Inches(11.5), Inches(0.9), title, size=32, bold=True)
    rect(s, Inches(0.7), Inches(1.28), Inches(0.9), Emu(38000), SIGNAL)
    foot = "Team F-28 · Distracted Driver Detection" + ("   ·   SMOKE-TEST NUMBERS, NOT FINAL" if SMOKE else "")
    text(s, Inches(0.7), Inches(7.0), Inches(9), Inches(0.3), foot, size=11, color=INK3)
    text(s, Inches(12.0), Inches(7.0), Inches(0.6), Inches(0.3), str(n), size=11, color=INK3, align=PP_ALIGN.RIGHT)
    if note:
        s.notes_slide.notes_text_frame.text = note
    return s


def picture(slide, path, x, y, w=None, h=None, card=True):
    if not Path(path).exists():
        text(slide, x, y, w or Inches(4), Inches(0.5), f"[missing figure: {Path(path).name}]", size=14, color=INK3)
        return
    if card:
        pad = Inches(0.08)
        pic = slide.shapes.add_picture(str(path), x + pad, y + pad, w - 2 * pad if w else None, h - 2 * pad if h else None)
        bg = rect(slide, x, y, pic.width + 2 * pad, pic.height + 2 * pad, RGBColor(255, 255, 255))
        slide.shapes._spTree.remove(bg._element); slide.shapes._spTree.insert(2, bg._element)
        return pic
    return slide.shapes.add_picture(str(path), x, y, w, h)


models = M["models"]
T = M["test"]
drop = M["drop"][DEEP]
sp = M["splits"]
cls_n = M["meta"]["classical_train_n"]
n_slide = 0


def nxt():
    global n_slide; n_slide += 1; return n_slide


# 1. Title
s = prs.slides.add_slide(BLANK); nxt()
s.background.fill.solid(); s.background.fill.fore_color.rgb = GROUND
text(s, Inches(0.9), Inches(2.0), Inches(11.5), Inches(1.8), "Machine Learning Techniques for Distracted Driver Detection", size=44, bold=True)
rect(s, Inches(0.9), Inches(3.75), Inches(1.2), Emu(50000), SIGNAL)
text(s, Inches(0.9), Inches(4.05), Inches(11), Inches(0.5), "Team F-28 · Project 75 · Section F · ML Mini Project", size=22, color=INK2)
text(s, Inches(0.9), Inches(4.75), Inches(11), Inches(1.0),
     ["Chiranth R (PES1UG24AM354)", "Vihaan Wadhwa (PES1UG24AM319)"], size=22)
if SMOKE:
    text(s, Inches(0.9), Inches(6.6), Inches(11), Inches(0.4), "Built from smoke-test numbers (1,000 images, 1 epoch). Not the final results.", size=14, color=RANDOM)
s.notes_slide.notes_text_frame.text = "Introduce the team and the project title."

# 2. Problem
s = new_slide("The problem", nxt(), "Explain the task: one photo in, one of ten activities out.")
bullets(s, Inches(0.7), Inches(1.7), Inches(6.3), Inches(4.8), [
    "Goal:|classify a dashboard-camera photo of a driver into one of 10 activities.",
    "Why it matters:|a system that notices texting, phone calls or reaching behind could warn the driver in time.",
    "Input:|one RGB photo taken from the passenger side.",
    "Output:|c0 safe driving, or one of nine distractions (c1–c9).",
    "Our question:|does the usual way of measuring accuracy hold up on drivers the model has never seen?",
], size=21)
names = M["class_names"]
y = Inches(1.75)
for c in M["classes"]:
    text(s, Inches(7.6), y, Inches(0.7), Inches(0.4), c, size=16, color=SIGNAL, bold=True)
    text(s, Inches(8.3), y, Inches(4.5), Inches(0.4), names[c], size=16, color=INK2)
    y += Inches(0.47)

# 3. Dataset
n_total = M["meta"]["n_images_used"]
s = new_slide("Dataset: State Farm (Kaggle)", nxt(),
              "All numbers here come from the labelled training set of the Kaggle competition.")
bullets(s, Inches(0.7), Inches(1.7), Inches(4.4), Inches(4.8), [
    f"{n_total:,} labelled photos|across 10 classes",
    f"{M['meta']['n_drivers']} different drivers|each appears in many photos",
    f"Classes are fairly balanced:|{min(M['class_counts'].values()):,} to {max(M['class_counts'].values()):,} photos per class",
    "Consecutive frames of the same person look very alike.|That's why a random split can leak.",
], size=19)
picture(s, FIG / "class_examples.png", Inches(5.4), Inches(1.65), w=Inches(7.3))

# 4. Reference paper
s = new_slide("Reference: Stanford CS229 (Spring 2019), project 24", nxt(),
              "We reproduce the paper's classical baselines and add a pretrained CNN.")
bullets(s, Inches(0.7), Inches(1.7), Inches(11.8), Inches(4.8), [
    "Features:|64×64×3 images flattened into 12,288 numbers.",
    "Models:|Naive Bayes, linear SVM, softmax regression, decision tree, 2-layer neural network.",
    "Evaluation:|a random 80/20 split of the photos; best accuracy reported around 92%.",
    "What we add:|(1) a driver-grouped split that keeps test drivers out of training, (2) a pretrained MobileNetV3, (3) Grad-CAM to see where models look.",
], size=24, gap=18)

# 5. Method: two splits
s = new_slide("Our method: two ways to split the same data", nxt(),
              "Both splits are 70/15/15 with seed 42. Only the grouping differs.")
for i, (key, title, colr, desc) in enumerate([
    ("random", "Random split", RANDOM, "Photos shuffled, stratified by class. The same drivers appear in train, val and test."),
    ("driver", "Driver split", DRIVER, "Whole drivers assigned to train, val or test. Test drivers are never seen in training."),
]):
    x = Inches(0.7 + i * 6.1)
    rect(s, x, Inches(1.7), Inches(5.8), Inches(4.6), PANEL)
    rect(s, x, Inches(1.7), Inches(5.8), Emu(60000), colr)
    text(s, x + Inches(0.35), Inches(1.95), Inches(5.2), Inches(0.5), title, size=26, bold=True)
    text(s, x + Inches(0.35), Inches(2.6), Inches(5.2), Inches(1.0), desc, size=17, color=INK2)
    yy = Inches(3.75)
    for part in ["train", "val", "test"]:
        p = sp[key][part]
        text(s, x + Inches(0.35), yy, Inches(1.2), Inches(0.5), part, size=18, color=INK3)
        text(s, x + Inches(1.5), yy, Inches(4), Inches(0.5), f"{p['n_images']:,} photos · {p['n_drivers']} drivers", size=18)
        yy += Inches(0.55)
text(s, Inches(0.7), Inches(6.45), Inches(12), Inches(0.5),
     "Driver-split test drivers: " + ", ".join(M["driver_split_drivers"]["test"]), size=15, color=INK3)

# 6. Models
meta = M["meta"]
s = new_slide("Models", nxt(), "Classical models follow the paper; MobileNetV3 is our deep model.")
rect(s, Inches(0.7), Inches(1.7), Inches(5.8), Inches(4.9), PANEL)
text(s, Inches(1.0), Inches(1.9), Inches(5.3), Inches(0.5), "Classical (like CS229)", size=24, bold=True)
bullets(s, Inches(1.0), Inches(2.6), Inches(5.3), Inches(4), [
    f"Input:|{meta['classical_img_size']}×{meta['classical_img_size']}×3 pixels, flattened, standardised (train only)",
    "Models:|Gaussian Naive Bayes, linear SVM (hinge SGD), softmax (multinomial logistic), decision tree",
    f"Training set:|subsample of {cls_n['random']:,} / {cls_n['driver']:,} photos (random / driver) for speed",
], size=17)
rect(s, Inches(6.8), Inches(1.7), Inches(5.8), Inches(4.9), PANEL)
text(s, Inches(7.1), Inches(1.9), Inches(5.3), Inches(0.5), "Deep: MobileNetV3-Large", size=24, bold=True)
bullets(s, Inches(7.1), Inches(2.6), Inches(5.3), Inches(4), [
    f"Input:|{meta['deep_img_size']}×{meta['deep_img_size']} RGB, ImageNet-pretrained weights",
    f"Training:|{meta['deep_epochs']} epochs, AdamW, one-cycle LR {meta['deep_lr']}, batch {meta['deep_batch_size']}, mixed precision on a {meta['gpu']}",
    "Augmentation:|small shifts, rotations, colour jitter. No horizontal flip: left/right-hand classes would swap.",
    "Model selection:|best epoch by validation accuracy, never the test set",
], size=17)

# 7. Results table
s = new_slide("Results: every model on both splits", nxt(), "Read the drop column: that's the cost of testing on new drivers.")
rows, cols = len(models) + 1, 6
tbl = s.shapes.add_table(rows, cols, Inches(0.7), Inches(1.8), Inches(11.9), Inches(0.6) * rows).table
heads = ["Model", "Random-split acc", "Driver-split acc", "Drop (points)", "Macro-F1 random", "Macro-F1 driver"]
for j, hd in enumerate(heads):
    c = tbl.cell(0, j); c.text = hd
for i, m in enumerate(models, start=1):
    vals = [m, pct(T["random"][m]["accuracy"]), pct(T["driver"][m]["accuracy"]), f"{M['drop'][m]['drop_points']:.1f}",
            f"{T['random'][m]['macro_f1']:.3f}", f"{T['driver'][m]['macro_f1']:.3f}"]
    for j, v in enumerate(vals):
        tbl.cell(i, j).text = v
for i in range(rows):
    for j in range(cols):
        c = tbl.cell(i, j); c.fill.solid()
        c.fill.fore_color.rgb = PANEL if i == 0 else (RGBColor(0x1D, 0x24, 0x30) if models[i - 1] == DEEP else GROUND) if i else PANEL
        for p in c.text_frame.paragraphs:
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
            for r in p.runs:
                r.font.size = Pt(17); r.font.name = FONT; r.font.bold = (i == 0 or j == 0)
                r.font.color.rgb = INK3 if i == 0 else INK
text(s, Inches(0.7), Inches(6.2), Inches(12), Inches(0.6),
     f"Test sets: {sp['random']['test']['n_images']:,} photos (random split) and {sp['driver']['test']['n_images']:,} photos from {len(M['driver_split_drivers']['test'])} unseen drivers (driver split).",
     size=15, color=INK3)

# 8. Headline chart
s = new_slide(f"Unseen drivers: MobileNetV3 {pct(drop['random_acc'])} → {pct(drop['driver_acc'])}", nxt(),
              "This is the main finding. Random splits measure partly how well the model recognises people it has already seen.")
cd = CategoryChartData(); cd.categories = models
cd.add_series("Random split (test drivers seen in training)", [T["random"][m]["accuracy"] for m in models])
cd.add_series("Driver split (test drivers never seen)", [T["driver"][m]["accuracy"] for m in models])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.7), Inches(1.6), Inches(11.9), Inches(5.2), cd)
ch = gf.chart
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False
ch.legend.font.color.rgb = INK2; ch.legend.font.size = Pt(15); ch.legend.font.name = FONT
for ser, colr in zip(ch.series, [RANDOM, DRIVER]):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = colr
    ser.data_labels.show_value = True; ser.data_labels.number_format = '0%'; ser.data_labels.number_format_is_linked = False
    ser.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    ser.data_labels.font.size = Pt(14); ser.data_labels.font.color.rgb = INK; ser.data_labels.font.name = FONT
plot = ch.plots[0]; plot.gap_width = 60; plot.overlap = -5
va = ch.value_axis; va.maximum_scale = 1.0; va.minimum_scale = 0; va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = RULE; va.tick_labels.number_format = '0%'; va.tick_labels.number_format_is_linked = False
va.tick_labels.font.color.rgb = INK3; va.tick_labels.font.size = Pt(13); va.format.line.fill.background()
ca = ch.category_axis; ca.tick_labels.font.color.rgb = INK2; ca.tick_labels.font.size = Pt(15); ca.format.line.color.rgb = RULE

# 9. Confusion matrix + hard classes
pc = T["driver"][DEEP]["per_class_accuracy"]
worst = sorted(pc.items(), key=lambda kv: kv[1])[:3]
s = new_slide("Where MobileNetV3 gets confused on new drivers", nxt(),
              "Rows are true classes, columns are predictions. Off-diagonal cells show mix-ups.")
picture(s, FIG / "cm_driver_mobilenetv3.png", Inches(0.7), Inches(1.55), h=Inches(5.3))
text(s, Inches(7.4), Inches(1.75), Inches(5.2), Inches(0.5), "Hardest classes (driver split)", size=22, bold=True)
bullets(s, Inches(7.4), Inches(2.45), Inches(5.2), Inches(3), [f"{c} {names[c]}:|{pct(v)} correct" for c, v in worst], size=19)
text(s, Inches(7.4), Inches(4.4), Inches(5.2), Inches(2),
     f"Per-class accuracy on the random split for the same classes: " + ", ".join(f"{c} {pct(T['random'][DEEP]['per_class_accuracy'][c])}" for c, _ in worst) + ".",
     size=17, color=INK2)

# 10. Grad-CAM
s = new_slide("Grad-CAM: where each model looks", nxt(),
              "Same unseen test photos for both models. Heatmap = regions that pushed the predicted class up.")
picture(s, FIG / "gradcam_grid.png", Inches(0.7), Inches(1.5), h=Inches(5.4))
gc = M.get("gradcam", [])
n_r = sum(r["pred_random"] == r["true"] for r in gc); n_d = sum(r["pred_driver"] == r["true"] for r in gc)
text(s, Inches(6.6), Inches(1.7), Inches(6.0), Inches(4.8), [
    f"On these {len(gc)} photos (neither model trained on them):",
    f"random-split model correct on {n_r}, driver-split model correct on {n_d}.",
    "",
    "Columns per photo: original · random-split model · driver-split model.",
    "Look for heat on hands, phone, cup and face, versus heat on the background or the person's appearance.",
], size=17, color=INK2)

# 10b. Cross-validation of the leakage finding and of debiasing (only if outputs/cv/cv_summary.json exists)
CV_PATH = OUT / "cv" / "cv_summary.json"
DB_PATH = OUT / "debias" / "metrics_debias.json"
if CV_PATH.exists():
    CV = json.loads(CV_PATH.read_text()); CVV = CV["variants"]; PV = CV["paired_vs_baseline"]; ID = CV["identity"]
    DB = json.loads(DB_PATH.read_text()) if DB_PATH.exists() else None
    base = CVV["baseline"]["test_acc"]
    s = new_slide(f"Is it real? 5-fold driver cross-validation ({CV['n_runs']} runs)", nxt(),
                  "Every one of the 26 drivers is an unseen test driver once; 3 seeds per fold. Each run picks its epoch on separate validation drivers. "
                  "p-values: paired t-test over the 5 folds (seeds averaged).")
    rows_v = ["baseline", "A", "C"]
    nm = {"baseline": "Baseline (light augmentation)", "A": "A: scramble appearance", "C": "C: A + adversary (λ 1.0)"}
    tbl = s.shapes.add_table(4, 4, Inches(0.7), Inches(1.75), Inches(7.4), Inches(0.62) * 4).table
    for j, wd in enumerate([3.2, 1.6, 1.4, 1.2]):
        tbl.columns[j].width = Inches(wd)
    for j, hd in enumerate(["Driver-split model", "Unseen-driver acc", "vs baseline", "p"]):
        tbl.cell(0, j).text = hd
    for i, v in enumerate(rows_v, start=1):
        t = CVV[v]["test_acc"]
        diff = "–" if v == "baseline" else f"{PV[v]['fold_level']['mean_diff']*100:+.1f} pts"
        pval = "–" if v == "baseline" else f"{PV[v]['fold_level']['t_test_p']:.2f}"
        for j, val in enumerate([nm[v], f"{t['mean']*100:.1f}% ± {t['std']*100:.1f}", diff, pval]):
            tbl.cell(i, j).text = val
    for i in range(4):
        for j in range(4):
            c = tbl.cell(i, j); c.fill.solid(); c.fill.fore_color.rgb = PANEL if i == 0 else GROUND
            for pp in c.text_frame.paragraphs:
                pp.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
                for r in pp.runs:
                    r.font.size = Pt(15); r.font.name = FONT; r.font.bold = (i == 0); r.font.color.rgb = INK3 if i == 0 else INK
    pc = PV["C"]["fold_level"]
    first = f"On our single split it looked like {pct(DB['variants']['baseline']['test']['accuracy'])} → {pct(DB['variants'][DB['selected_by_validation']]['test']['accuracy'])}. " if DB else ""
    bullets(s, Inches(8.4), Inches(1.75), Inches(4.3), Inches(5), [
        f"The leak is robust:|random split {pct(T['random'][DEEP]['accuracy'])} vs {base['mean']*100:.1f}% ± {base['std']*100:.1f} on unseen drivers, averaged over all 26 drivers.",
        f"Debiasing helps only a little:|{first}Across all drivers C is {pc['mean_diff']*100:+.1f} pts (95% CI {pc['ci95_t'][0]*100:+.1f} to {pc['ci95_t'][1]*100:+.1f}), better on {pc['folds_improved']}/{pc['n_folds']} folds, not significant (p = {pc['t_test_p']:.2f}).",
        "Lesson:|one split with 4 test drivers can mislead by several points.",
    ], size=15, gap=10)
    picture(s, OUT / "cv" / "figures" / "cv_per_fold.png", Inches(0.7), Inches(4.45), h=Inches(2.45))

    idc = PV["C"]["one_shot_id"]
    s = new_slide("Did the adversary remove 'who is this?'", nxt(),
                  "One-shot identification: within one action, match each photo to one reference photo per test driver using the model's features. "
                  "Chance is about 1 in 5. Lower means less identity information.")
    picture(s, OUT / "cv" / "figures" / "cv_identity.png", Inches(0.7), Inches(1.9), w=Inches(7.6))
    bullets(s, Inches(8.6), Inches(1.7), Inches(4.1), Inches(5), [
        f"ImageNet features identify people {ID['imagenet']['one_shot_id']['mean']:.0%} of the time|from one photo, even within the same action.",
        f"Fine-tuning on actions lowers it to about {ID['baseline']['one_shot_id']['mean']:.0%}.|",
        f"The adversary (C) lowers it further, to {ID['C']['one_shot_id']['mean']:.0%}:|lower on all {idc['n_folds']} folds, p = {idc['t_test_p']:.3f}.",
        "But identity stays far above chance,|and the accuracy gain is small, so appearance is only part of the problem.",
    ], size=15, gap=10)

# 11. Limitations
s = new_slide("Limitations", nxt())
bullets(s, Inches(0.7), Inches(1.7), Inches(11.8), Inches(5), [
    f"Few drivers:|only {M['meta']['n_drivers']} people, and {len(M['driver_split_drivers']['test'])} in the driver-split test set, so per-driver results vary a lot.",
    "One camera setup:|same car, angle and lighting style; a new car or camera would be another shift.",
    f"Classical models used a {cls_n['random']:,}-photo subsample|for speed.",
    "One seed, one split of drivers, no hyperparameter search:|numbers would move somewhat with a different driver assignment.",
    "Grad-CAM is a rough explanation,|not proof of what the model reasons about.",
], size=23, gap=16)

# 12. Future work + conclusion
s = new_slide("Future work and conclusion", nxt())
bullets(s, Inches(0.7), Inches(1.7), Inches(6.0), Inches(5), [
    "Driver-grouped cross-validation|to average over many driver assignments.",
    "More drivers and cameras|for a fairer test of generalisation.",
    "Next research step:|test vision-language models (CLIP) with driver-appearance decoupling under realistic calibration.",
], size=19)
rect(s, Inches(7.1), Inches(1.7), Inches(5.5), Inches(4.8), PANEL)
text(s, Inches(7.4), Inches(1.95), Inches(5.0), Inches(0.5), "Conclusion", size=24, bold=True)
best_r = max(models, key=lambda m: T["random"][m]["accuracy"])
text(s, Inches(7.4), Inches(2.6), Inches(5.0), Inches(3.8), [
    f"On a random split, {best_r} reaches {pct(T['random'][best_r]['accuracy'])}.",
    f"On unseen drivers, MobileNetV3 scores {pct(drop['driver_acc'])} ({abs(drop['drop_points']):.1f} points {'lower' if drop['drop_points'] > 0 else 'higher'} than on the random split).",
    "Random splits overstate how well distracted-driver models work on new people. Report driver-grouped results.",
], size=18, color=INK2)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
prs.save(args.out)
print(f"saved {args.out} with {len(prs.slides)} slides (smoke={SMOKE})")
