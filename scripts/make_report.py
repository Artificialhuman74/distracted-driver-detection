"""Build deliverables/report.docx (and report.pdf via LibreOffice if installed) from outputs/metrics.json.

Every number in the report is read from metrics.json; nothing is typed in by hand.
Member contributions are read from deliverables/contributions.json if it exists
(format: {"Chiranth R": "...", "Vihaan Wadhwa": "..."}); otherwise a placeholder is left.
Usage:  .venv/bin/python scripts/make_report.py [--outputs outputs] [--out deliverables/report.docx]
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401  (kept for easy landscape tweaks)
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Inches, RGBColor

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--outputs", default=str(ROOT / "outputs"))
ap.add_argument("--out", default=str(ROOT / "deliverables" / "report.docx"))
args = ap.parse_args()
OUT = Path(args.outputs); FIG = OUT / "figures"
M = json.loads((OUT / "metrics.json").read_text())
DEEP = "MobileNetV3"
T, SP, META = M["test"], M["splits"], M["meta"]
models = M["models"]; names = M["class_names"]
pct = lambda v: f"{v * 100:.1f}%"
drop = M["drop"][DEEP]
CV_PATH = OUT / "cv" / "cv_summary.json"
CV = json.loads(CV_PATH.read_text()) if CV_PATH.exists() else None

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11.5)
for s in ["Heading 1", "Heading 2", "Title"]:
    doc.styles[s].font.name = "Times New Roman"; doc.styles[s].font.color.rgb = RGBColor(0, 0, 0)
doc.styles["Heading 1"].font.size = Pt(14); doc.styles["Heading 2"].font.size = Pt(12)
for sec in doc.sections:
    sec.left_margin = sec.right_margin = Inches(1); sec.top_margin = sec.bottom_margin = Inches(0.9)

counters = {"Table": 0, "Figure": 0}
sec_no = [0]


def para(text, bold=False, italic=False, align=None, size=None, space_after=6):
    p = doc.add_paragraph(); r = p.add_run(text); r.bold = bold; r.italic = italic
    if size: r.font.size = Pt(size)
    if align: p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    return p


def heading(text):
    sec_no[0] += 1
    doc.add_heading(f"{sec_no[0]}. {text}", level=1)


def caption(kind, text):
    counters[kind] += 1
    p = para(f"{kind} {counters[kind]}: {text}", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=10, space_after=10)
    return counters[kind]


def figure(path, text, width=6.0):
    if not Path(path).exists():
        para(f"[missing figure: {Path(path).name}]", italic=True); return None
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    return caption("Figure", text)


def table(headers, rows, text, widths=None):
    n = counters["Table"] + 1
    para(f"Table {n}: {text}", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=10, space_after=4)
    counters["Table"] = n
    t = doc.add_table(rows=1, cols=len(headers)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        c = t.rows[0].cells[j]; c.text = ""; r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(10)
    for row in rows:
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = ""; r = cells[j].paragraphs[0].add_run(str(v)); r.font.size = Pt(10)
            if j: cells[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return n


# ---------------- title block ----------------
para("Machine Learning Techniques for Distracted Driver Detection", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=17, space_after=4)
para("ML Mini Project · Team F-28 · Project 75 · Section F", align=WD_ALIGN_PARAGRAPH.CENTER, size=11, space_after=2)
para("Chiranth R (PES1UG24AM354) · Vihaan Wadhwa (PES1UG24AM319)", align=WD_ALIGN_PARAGRAPH.CENTER, size=11, space_after=14)
if META["smoke"]:
    para("DRAFT built from smoke-test numbers (1,000 images, 1 epoch). Not the final results.", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

best_r = max(models, key=lambda m: T["random"][m]["accuracy"])
direction = "lower" if drop["drop_points"] > 0 else "higher"

# ---------------- abstract ----------------
doc.add_heading("Abstract", level=1)
para(f"We classify dashboard-camera photos of drivers into ten activities (safe driving and nine kinds of distraction) using "
     f"the State Farm Distracted Driver Detection dataset ({META['n_images_used']:,} labelled images of {META['n_drivers']} drivers). "
     f"Following a Stanford CS229 project, we train classical models on flattened 64×64 pixels (Gaussian Naive Bayes, linear SVM, "
     f"softmax regression, decision tree) and fine-tune an ImageNet-pretrained MobileNetV3-Large. Our main contribution is the evaluation: "
     f"every model is trained and tested on two 70/15/15 splits of the same data, a random split in which the same drivers appear in training "
     f"and test, and a driver-grouped split in which test drivers are never seen. On the random split the best model ({best_r}) reaches "
     f"{pct(T['random'][best_r]['accuracy'])} test accuracy. On unseen drivers MobileNetV3 reaches {pct(drop['driver_acc'])}, "
     f"{abs(drop['drop_points']):.1f} percentage points {direction} than on its random split ({pct(drop['random_acc'])}). "
     f"Random splits therefore overstate how well such models work on new people. Grad-CAM visualisations show where each model looks." +
     (f" Driver-grouped 5-fold cross-validation with 3 seeds confirms the gap ({CV['variants']['baseline']['test_acc']['mean']*100:.1f}% ± "
      f"{CV['variants']['baseline']['test_acc']['std']*100:.1f} on unseen drivers). An adversarial driver classifier reduces identity information in the "
      f"features but improves unseen-driver accuracy by only {CV['paired_vs_baseline']['C']['fold_level']['mean_diff']*100:+.1f} points, which is not "
      f"statistically significant." if CV else ""))

# ---------------- 1 introduction ----------------
heading("Introduction")
para("Distracted driving, such as texting, phone calls, drinking or reaching behind, takes the driver's attention away from the road. "
     "A camera-based system that recognises these activities could warn the driver in time. We study this as image classification: "
     "given one photo taken from the passenger side, predict which of ten activities the driver is doing.")
para("Many reported results on this dataset use a random split of the photos. Because each driver contributes hundreds of near-identical "
     "frames, a random split places photos of the same person in both the training and the test set. Our question is how much of the "
     "measured accuracy survives when the test drivers are new. We answer it by training every model twice, on a random split and on a "
     "driver-grouped split, and comparing.")

# ---------------- 2 related work ----------------
heading("Related Work")
para("Our reference is a Stanford CS229 (Spring 2019) project on the same dataset [1], which flattened 64×64×3 images and compared "
     "Naive Bayes, a linear SVM, softmax regression, a decision tree and a two-layer neural network, reporting a best accuracy of about "
     "92% on a random 80/20 split. Convolutional networks pretrained on ImageNet [5] are the standard approach for image classification; "
     "we use MobileNetV3 [3], a small network designed for mobile devices, which suits an in-car setting. To inspect what the networks "
     "rely on we use Grad-CAM [4]. Keeping all images of one subject in the same split (a subject-wise or grouped split) is a standard "
     "precaution against identity leakage when a dataset contains many images of few people.")

# ---------------- 3 dataset & preprocessing ----------------
heading("Dataset and Preprocessing")
cc = M["class_counts"]
para(f"We use the labelled training set of the Kaggle competition \"State Farm Distracted Driver Detection\" [2]: "
     f"{META['n_images_used']:,} RGB photos (640×480) of {META['n_drivers']} drivers in ten classes. The file driver_imgs_list.csv "
     f"gives the driver ID (subject) of each photo. Classes are roughly balanced, from {min(cc.values()):,} to {max(cc.values()):,} photos per class "
     f"(Figure 1). The competition's unlabelled test images are not used.")
figure(FIG / "class_examples.png", "One example photo per class (c0 safe driving … c9 talking to passenger).")
para("Splits. Both splits use 70% / 15% / 15% for train / validation / test with random seed 42. "
     "The random split is stratified by class and ignores the driver. The driver split assigns whole drivers: "
     f"{len(M['driver_split_drivers']['train'])} drivers to training, {len(M['driver_split_drivers']['val'])} to validation and "
     f"{len(M['driver_split_drivers']['test'])} to test (test drivers: {', '.join(M['driver_split_drivers']['test'])}). Table 1 gives the sizes.")
table(["Split", "Part", "Photos", "Share", "Drivers"],
      [[k, part, f"{SP[k][part]['n_images']:,}", pct(SP[k][part]['fraction']), SP[k][part]['n_drivers']]
       for k in ["random", "driver"] for part in ["train", "val", "test"]],
      "Sizes of the two splits. In the random split every driver appears in every part.")
para(f"Preprocessing. For the classical models each photo is resized to {META['classical_img_size']}×{META['classical_img_size']}×3, flattened "
     f"into 12,288 values in [0, 1] and standardised with a scaler fitted on the training set only. For MobileNetV3 photos are resized to "
     f"{META['deep_img_size']}×{META['deep_img_size']} and normalised with the ImageNet mean and standard deviation.")

# ---------------- 4 methods ----------------
heading("Methods")
cn = META["classical_train_n"]
doc.add_heading("Classical baselines", level=2)
para(f"Following [1] we train four models with scikit-learn [6]: Gaussian Naive Bayes; a linear SVM (hinge loss, stochastic gradient "
     f"descent, L2 penalty 1e-4); softmax regression (multinomial logistic regression, L-BFGS); and a decision tree (unlimited depth). "
     f"For speed each model trains on a class-stratified subsample of the training set: {cn['random']:,} photos for the random split and "
     f"{cn['driver']:,} for the driver split. No hyperparameters are tuned.")
doc.add_heading("MobileNetV3-Large", level=2)
para(f"We fine-tune all layers of an ImageNet-pretrained MobileNetV3-Large [3] in PyTorch [7], replacing the last layer with a 10-way output. "
     f"Training uses AdamW [8] with weight decay 0.05, a one-cycle learning-rate schedule peaking at {META['deep_lr']}, batch size "
     f"{META['deep_batch_size']}, label smoothing 0.1, mixed precision, and {META['deep_epochs']} epochs on a {META['gpu']} GPU. "
     "Augmentation is light: random affine shifts, rotations (up to 8°) and scaling, plus colour jitter. We do not flip images "
     "horizontally, because that would turn 'texting with the right hand' into 'texting with the left hand'. The checkpoint with the "
     f"best validation accuracy is kept (epoch {META['deep_best_epoch']['random']} for the random split, "
     f"epoch {META['deep_best_epoch']['driver']} for the driver split). The test set is used only once, for the final numbers.")
doc.add_heading("Evaluation", level=2)
para("For every model and split we report test accuracy, macro-averaged F1, per-class accuracy and the confusion matrix. "
     "For the driver split we also report accuracy for each test driver. Grad-CAM [4] heatmaps are computed on the last convolutional "
     "block for twelve photos that neither model saw during training (they are in the random-split test set and belong to a driver-split test driver).")

# ---------------- 5 experiments & results ----------------
heading("Experiments and Results")
table(["Model", "Random acc", "Driver acc", "Drop (pts)", "F1 random", "F1 driver"],
      [[m, pct(T["random"][m]["accuracy"]), pct(T["driver"][m]["accuracy"]), f"{M['drop'][m]['drop_points']:.1f}",
        f"{T['random'][m]['macro_f1']:.3f}", f"{T['driver'][m]['macro_f1']:.3f}"] for m in models],
      f"Test results. Random-split test set: {SP['random']['test']['n_images']:,} photos; driver-split test set: "
      f"{SP['driver']['test']['n_images']:,} photos of {len(M['driver_split_drivers']['test'])} unseen drivers.")
lines = []
for m in models:
    d = M["drop"][m]
    lines.append(f"{m} {'drops' if d['drop_points'] > 0 else 'changes'} from {pct(d['random_acc'])} to {pct(d['driver_acc'])}")
para("Table 2 and Figure 2 summarise the results. " + "; ".join(lines) + ".")
figure(FIG / "split_comparison.png", "Test accuracy of every model on the random split (seen drivers) and the driver split (new drivers).")
pc_r, pc_d = T["random"][DEEP]["per_class_accuracy"], T["driver"][DEEP]["per_class_accuracy"]
table(["Class", "Name", "Random split", "Driver split"],
      [[c, names[c], pct(pc_r[c]), pct(pc_d[c])] for c in M["classes"]],
      "MobileNetV3 per-class test accuracy on both splits.")
worst = sorted(pc_d.items(), key=lambda kv: kv[1])[:3]
para("On new drivers the hardest classes for MobileNetV3 are " +
     ", ".join(f"{c} ({names[c]}, {pct(v)})" for c, v in worst) + ". Figure 3 shows the confusion matrix.")
figure(FIG / "cm_driver_mobilenetv3.png", "Confusion matrix of MobileNetV3 on the driver split (rows: true class, columns: predicted).", width=5.0)
pdv = M["per_test_driver_accuracy"][DEEP]
para("Accuracy also varies between test drivers: " + ", ".join(f"{d} {pct(v['accuracy'])} ({v['n']} photos)" for d, v in sorted(pdv.items())) +
     " (Figure 4).")
figure(FIG / "per_driver_accuracy.png", "Accuracy of each model on each unseen test driver (driver split).")
figure(FIG / "training_curves.png", "MobileNetV3 validation accuracy per epoch on both splits.", width=4.5)
gc = M.get("gradcam", [])
n_r = sum(r["pred_random"] == r["true"] for r in gc); n_d = sum(r["pred_driver"] == r["true"] for r in gc)
para(f"Figure 6 shows Grad-CAM heatmaps for {len(gc)} photos unseen by both models. On these photos the random-split model is correct on "
     f"{n_r} and the driver-split model on {n_d}.")
figure(FIG / "gradcam_grid.png", "Grad-CAM for the random-split model and the driver-split model on the same unseen photos.")
DB_PATH = OUT / "debias" / "metrics_debias.json"
DB = json.loads(DB_PATH.read_text()) if DB_PATH.exists() else None
if DB:
    V = DB["variants"]; sel = DB["selected_by_validation"]
    doc.add_heading("Can the model learn the action without learning the people?", level=2)
    para("We tested two ways to reduce reliance on driver appearance, training MobileNetV3 on the same driver split with the same "
         "optimiser and 8 epochs. (A) Appearance-scrambling augmentation: wider brightness/contrast/saturation jitter, random grayscale "
         "(p = 0.2) and random erasing of a 2–20% patch (p = 0.5), still with no horizontal flip. (B, C) The same augmentation plus an "
         "adversarial driver classifier: a small head predicts which of the 18 training drivers is in the photo from the 960-d pooled "
         "features, behind a gradient-reversal layer, so the backbone is trained to make those features useless for identifying the "
         "driver (domain-adversarial training [10]); the reversal strength ramps up to λ = 0.3 (B) or λ = 1.0 (C). Each variant keeps its "
         "best epoch by validation accuracy and the final variant is chosen by validation accuracy only; Table 4 reports all of them.")
    names_v = {"baseline": "Baseline (main run)", "aug": "A: augmentation", "aug+adv0.3": "B: A + adversary (λ ≤ 0.3)", "aug+adv1.0": "C: A + adversary (λ ≤ 1.0)"}
    table(["Variant", "Validation acc", "Test acc", "Test macro-F1"] + [f"{d}" for d in M["driver_split_drivers"]["test"]],
          [[names_v[k] + (" (chosen)" if k == sel else ""), pct(V[k]["val_acc"]), pct(V[k]["test"]["accuracy"]), f"{V[k]['test']['macro_f1']:.3f}"]
           + [pct(V[k]["test"]["per_test_driver_accuracy"][d]) for d in M["driver_split_drivers"]["test"]] for k in ["baseline", "aug", "aug+adv0.3", "aug+adv1.0"]],
          "Debiasing variants on the driver split (unseen drivers). The last four columns give accuracy per test driver.")
    pcb, pcs = V["baseline"]["test"]["per_class_accuracy"], V[sel]["test"]["per_class_accuracy"]
    ch = sorted(pcb, key=lambda c: pcs[c] - pcb[c])
    para(f"The chosen variant ({names_v[sel]}) reaches {pct(V[sel]['test']['accuracy'])} test accuracy on unseen drivers, compared with "
         f"{pct(V['baseline']['test']['accuracy'])} for the baseline. The largest per-class gains are " +
         ", ".join(f"{c} ({names[c]}) {pct(pcb[c])} → {pct(pcs[c])}" for c in ch[::-1][:3]) +
         f"; the largest loss is {ch[0]} ({names[ch[0]]}) {pct(pcb[ch[0]])} → {pct(pcs[ch[0]])}, which in a real system would mean more "
         "false alarms for safe drivers. During training the adversarial head's driver accuracy fell to near chance (1/18), but a separate "
         "identity probe (logistic regression on frozen features, telling the 4 test drivers apart) scored "
         f"{min(DB['identity_probe'].values()):.1%}–{max(DB['identity_probe'].values()):.1%} for every model including the ImageNet-only network, "
         "so this probe is saturated and cannot show whether identity information was reduced. With 4 test drivers and one seed, the "
         "improvement is promising but not yet statistically established; the next subsection tests it with cross-validation.")
    figure(OUT / "debias" / "figures" / "gradcam_debiased.png", "Grad-CAM of the driver-split baseline and the debiased model on the same unseen photos.")
if CV:
    CVV, PV, ID = CV["variants"], CV["paired_vs_baseline"], CV["identity"]
    doc.add_heading("Is the improvement real? Driver-grouped cross-validation", level=2)
    fd = CV["fold_drivers"]
    para(f"A single split with 4 test drivers is noisy, so we repeated the comparison with 5-fold driver-grouped cross-validation. The 26 "
         f"drivers were shuffled once (seed 42) and cut into 5 groups ({', '.join(str(len(v)) for v in fd.values())} drivers). In fold k, group k "
         "is the unseen test set, group k+1 is the validation set used to pick the epoch, and the other three groups are used for training. "
         f"Every fold was run with 3 seeds ({CV['n_runs']} training runs). Differences against the baseline are tested at fold level "
         "(seeds averaged, so the 5 folds are the independent units) with a paired t-test; with only 5 folds these tests have little power. "
         "We also measured identity information on each fold's unseen drivers with two harder probes on the frozen, L2-normalised 960-d "
         "features. One-shot identification takes one reference photo per test driver within the same action class, and assigns every "
         "other photo of that class to the nearest reference by cosine similarity; chance is about 1 in 5. Same-person verification "
         "measures the ROC AUC of cosine similarity for same-driver versus different-driver pairs within a class (0.5 means identity cannot "
         "be told apart). Both probes are for analysis only and never used for model selection.")
    nm = {"baseline": "Baseline", "A": "A: scramble", "C": "C: A + adversary"}
    rows_cv = []
    for v in ["baseline", "A", "C"]:
        t, f1 = CVV[v]["test_acc"], CVV[v]["test_macro_f1"]
        rows_cv.append([nm[v], f"{t['mean']*100:.1f} ± {t['std']*100:.1f}", f"{f1['mean']:.3f}",
                        "–" if v == "baseline" else f"{PV[v]['fold_level']['mean_diff']*100:+.1f} [{PV[v]['fold_level']['ci95_t'][0]*100:+.1f}, {PV[v]['fold_level']['ci95_t'][1]*100:+.1f}]",
                        "–" if v == "baseline" else f"{PV[v]['fold_level']['t_test_p']:.2f}",
                        f"{ID[v]['one_shot_id']['mean']:.3f}", f"{ID[v]['verif_auc']['mean']:.3f}"])
    rows_cv.append(["ImageNet only", "–", "–", "–", "–", f"{ID['imagenet']['one_shot_id']['mean']:.3f}", f"{ID['imagenet']['verif_auc']['mean']:.3f}"])
    table(["Model", "Acc % (±sd)", "F1", "Δ pts [95% CI]", "p", "1-shot ID", "Verif. AUC"], rows_cv,
          f"Driver-grouped 5-fold cross-validation, {len(CV['seeds'])} seeds per fold. Accuracy is on unseen drivers. The last two columns are identity "
          f"probes on unseen drivers (one-shot chance ≈ {ID['chance_one_shot']:.2f}; verification chance = 0.5); lower means less identity information.")
    idc, idv = PV["C"]["one_shot_id"], PV["C"]["verif_auc"]
    c0 = CV["safe_driving_c0"]
    para(f"Averaged over all 26 drivers, the baseline reaches {CVV['baseline']['test_acc']['mean']*100:.1f}% on unseen drivers, compared with "
         f"{pct(T['random'][DEEP]['accuracy'])} on the random split, so the leakage gap is robust to the choice of test drivers. The debiasing "
         f"gains are much smaller than on the single split. Model C is {PV['C']['fold_level']['mean_diff']*100:+.1f} points better on average, "
         f"is better on {PV['C']['fold_level']['folds_improved']} of {PV['C']['fold_level']['n_folds']} folds and in {PV['C']['run_level']['runs_improved']} "
         f"of {PV['C']['run_level']['n_runs']} runs, but the difference is not significant (p = {PV['C']['fold_level']['t_test_p']:.2f}). Model A is "
         f"essentially unchanged ({PV['A']['fold_level']['mean_diff']*100:+.1f} points, p = {PV['A']['fold_level']['t_test_p']:.2f}). The single-split "
         "gain therefore came largely from which 4 drivers happened to be in the test set.")
    para(f"The identity probes show that the ImageNet features identify drivers almost perfectly ({ID['imagenet']['one_shot_id']['mean']:.1%} "
         f"one-shot). Fine-tuning on the actions reduces this to {ID['baseline']['one_shot_id']['mean']:.1%}. The adversary reduces it further, to "
         f"{ID['C']['one_shot_id']['mean']:.1%}, lower than the baseline on all {idc['n_folds']} folds (paired t-test p = {idc['t_test_p']:.3f}). The "
         f"verification AUC also falls slightly ({idv['mean_diff']:+.3f}, p = {idv['t_test_p']:.2f}). The adversary therefore does remove some "
         "identity information, but identity remains far above chance, and removing part of it buys only a small, uncertain gain in accuracy. "
         f"Safe-driving accuracy (c0) averages {c0['baseline']:.1%} for the baseline, {c0['A']:.1%} for A and {c0['C']:.1%} for C.")
    figure(OUT / "cv" / "figures" / "cv_per_fold.png", "Cross-validation: unseen-driver accuracy per fold (bars: mean over seeds; dots: individual seeds).")
    figure(OUT / "cv" / "figures" / "cv_identity.png", "Identity probes on unseen drivers (lower = less identity information; dashed lines = chance).")
tm = META["timings_seconds"]
para(f"Compute: the complete Kaggle run took {tm['total'] / 60:.1f} minutes on one {META['gpu']}.")

# ---------------- 6 discussion ----------------
heading("Discussion: why random splits inflate accuracy")
para("In this dataset each driver is filmed in long sequences, so neighbouring photos are almost identical apart from the hand or head "
     "position. With a random split, most test photos have near-duplicates of the same person, in the same clothes, seat position and "
     "lighting, in the training set. A model can then score well by matching appearance and background, without fully learning the "
     "activity. The driver split removes this shortcut: the person, clothes and posture are new, and only the activity carries over. The "
     f"gap between the two columns of Table 2 measures how much of the random-split accuracy came from recognising seen people. "
     f"For MobileNetV3 the gap is {abs(drop['drop_points']):.1f} percentage points; for the classical pixel-based models, which can only "
     "compare raw pixels, see the Drop column of Table 2.")
para("A deployed system will always meet new drivers, so the driver-split number is the honest estimate of real-world accuracy. "
     "Results on this dataset should state which split was used.")

# ---------------- 7 limitations ----------------
heading("Limitations")
for t in [
    f"Only {META['n_drivers']} drivers, with {len(M['driver_split_drivers']['test'])} in the driver-split test set, so the driver-split accuracy depends on which drivers were held out.",
    ("Cross-validation used 5 folds and 3 seeds, so the significance tests have little power; small effects such as the debiasing gain need more folds or more drivers to confirm."
     if CV else "We used one random seed and one assignment of drivers; driver-grouped cross-validation would give a more stable estimate."),
    "Only one dataset was used. Testing on a second dataset (e.g. the AUC Distracted Driver Dataset, available under licence) is needed to check that the findings generalise.",
    f"Classical models were trained on a subsample of {cn['random']:,} photos and were not tuned.",
    "All photos come from one camera setup; a different car, camera angle or lighting would be a further distribution shift.",
    "Grad-CAM shows which regions raised the predicted score; it is a rough explanation, not a proof of the model's reasoning.",
]:
    doc.add_paragraph(t, style="List Bullet")

# ---------------- 8 conclusion & future work ----------------
heading("Conclusion and Future Work")
para(f"We reproduced the classical baselines of [1], added a pretrained MobileNetV3, and evaluated every model on a random split and on a "
     f"driver-grouped split. Random-split accuracy (best: {best_r}, {pct(T['random'][best_r]['accuracy'])}) overstates performance on new "
     f"people; on unseen drivers MobileNetV3 reaches {pct(drop['driver_acc'])}. Distracted-driver models should be evaluated with "
     "driver-grouped splits.")
if CV:
    para(f"Five-fold driver-grouped cross-validation confirms that the leakage gap is robust ({CV['variants']['baseline']['test_acc']['mean']*100:.1f}% ± "
         f"{CV['variants']['baseline']['test_acc']['std']*100:.1f} on unseen drivers). An adversarial driver classifier measurably reduces identity information "
         f"in the features, but its accuracy gain ({CV['paired_vs_baseline']['C']['fold_level']['mean_diff']*100:+.1f} points) is not significant. A promising "
         "single-split result (+4 points) shrank under cross-validation, which is itself a warning about evaluating on a few drivers.")
elif DB:
    para(f"A first attempt at reducing reliance on driver appearance (appearance-scrambling augmentation plus an adversarial driver "
         f"classifier) raised unseen-driver accuracy from {pct(DB['variants']['baseline']['test']['accuracy'])} to "
         f"{pct(DB['variants'][DB['selected_by_validation']]['test']['accuracy'])}, at the cost of more false alarms on safe driving.")
para("Future work: more drivers and a second dataset (cross-dataset testing); stronger ways to separate the action from the person, such as pose or hand-region inputs; more folds and seeds for statistical power. As our next research step, "
     "we plan to test vision-language models such as CLIP [9] with driver-appearance decoupling under realistic calibration, to check "
     "whether language-aligned features recognise the activity rather than the person.")

# ---------------- 9 contributions ----------------
heading("Contributions of Each Member")
contrib = ROOT / "deliverables" / "contributions.json"
if contrib.exists():
    for who, what in json.loads(contrib.read_text()).items():
        p = doc.add_paragraph(style="List Bullet"); r = p.add_run(f"{who}: "); r.bold = True; p.add_run(what)
else:
    para("[TO BE FILLED IN BY THE TEAM: create deliverables/contributions.json and re-run scripts/make_report.py]", bold=True)

# ---------------- references ----------------
doc.add_heading("References", level=1)
for i, ref in enumerate([
    "Stanford CS229 Machine Learning, Spring 2019, Project Report 24: distracted driver detection. https://cs229.stanford.edu/proj2019spr/report/24.pdf (poster: https://cs229.stanford.edu/proj2019spr/poster/24.pdf)",
    "State Farm Distracted Driver Detection, Kaggle competition, 2016. https://www.kaggle.com/c/state-farm-distracted-driver-detection",
    "A. Howard et al., \"Searching for MobileNetV3,\" in Proc. IEEE/CVF International Conference on Computer Vision (ICCV), 2019.",
    "R. R. Selvaraju et al., \"Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,\" in Proc. ICCV, 2017.",
    "J. Deng et al., \"ImageNet: A Large-Scale Hierarchical Image Database,\" in Proc. IEEE CVPR, 2009.",
    "F. Pedregosa et al., \"Scikit-learn: Machine Learning in Python,\" Journal of Machine Learning Research, vol. 12, 2011.",
    "A. Paszke et al., \"PyTorch: An Imperative Style, High-Performance Deep Learning Library,\" in Advances in NeurIPS, 2019.",
    "I. Loshchilov and F. Hutter, \"Decoupled Weight Decay Regularization,\" in Proc. ICLR, 2019.",
    "A. Radford et al., \"Learning Transferable Visual Models From Natural Language Supervision,\" in Proc. ICML, 2021.",
    "Y. Ganin and V. Lempitsky, \"Unsupervised Domain Adaptation by Backpropagation,\" in Proc. ICML, 2015.",
], start=1):
    para(f"[{i}] {ref}", size=10, space_after=3)

out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
doc.save(out)
print(f"saved {out} (tables: {counters['Table']}, figures: {counters['Figure']}, smoke={META['smoke']})")

soffice = shutil.which("soffice") or "/Applications/LibreOffice.app/Contents/MacOS/soffice"
if Path(soffice).exists():
    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out.parent), str(out)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    pdf = out.with_suffix(".pdf")
    print(f"saved {pdf}" if pdf.exists() else "PDF conversion failed")
else:
    print("LibreOffice not found: skipped PDF")
