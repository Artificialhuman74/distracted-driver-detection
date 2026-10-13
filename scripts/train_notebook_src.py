# %% [markdown]
# # Machine Learning Techniques for Distracted Driver Detection
# **Team F-28 · Project 75 · Section F** — Chiranth R (PES1UG24AM354), Vihaan Wadhwa (PES1UG24AM319)
#
# Data: Kaggle "State Farm Distracted Driver Detection" (22,424 labelled images, 10 classes, 26 drivers).
#
# **Main question:** a *random* train/test split puts the SAME drivers in train and test.
# Does accuracy drop when we test on drivers the model has never seen (a *driver* split)?
#
# What this notebook does:
# 1. Builds two 70/15/15 splits (random and driver-grouped), seed 42.
# 2. Trains classical baselines like the CS229 paper (64x64x3 flattened pixels).
# 3. Fine-tunes MobileNetV3-Large (ImageNet-pretrained) on both splits.
# 4. Saves every number to `metrics.json`, plus figures, models, Grad-CAM and demo images.

# %%
# ----------------------------- SETTINGS -----------------------------
SMOKE = True                 # True = quick bug check (~1,000 real images, 1 epoch). False = full run.
SEED = 42
SMOKE_N_IMAGES = 1000
CLASSICAL_MAX_TRAIN = 8000   # classical models use at most this many training images (for speed)
TREE_MAX_DEPTH = 20          # decision tree depth limit (for speed)
DEEP_EPOCHS = 1 if SMOKE else 8
BATCH_SIZE = 64
LR = 1e-3
IMG_DEEP = 224               # MobileNetV3 input size
IMG_SMALL = 64               # classical models input size (like the CS229 paper)
N_GRADCAM = 12
N_SAMPLES_PER_CLASS = 3      # 3 x 10 classes = 30 demo images

import os, time, json, random, shutil, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

T_START = time.time()
TIMINGS = {}  # stage name -> seconds


class Stage:
    """Tiny timer: `with Stage("name"):` prints how long a step took."""
    def __init__(self, name): self.name = name
    def __enter__(self):
        self.t = time.time(); print(f"\n===== {self.name} =====", flush=True)
    def __exit__(self, *a):
        dt = time.time() - self.t
        TIMINGS[self.name] = round(dt, 1)
        print(f"----- {self.name} took {dt/60:.1f} min (total so far {(time.time()-T_START)/60:.1f} min)", flush=True)


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    try:
        import torch; torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    except ImportError:
        pass

set_seed()

# Kaggle saves to /kaggle/working/outputs; anywhere else (e.g. Colab) saves to ./outputs
OUT = Path("/kaggle/working/outputs") if Path("/kaggle/working").exists() else Path("outputs").resolve()
FIG = OUT / "figures"
for d in [OUT, FIG, OUT / "splits", OUT / "sample_images"]:
    d.mkdir(parents=True, exist_ok=True)

CLASSES = [f"c{i}" for i in range(10)]
CLASS_NAMES = {
    "c0": "safe driving", "c1": "texting - right", "c2": "talking on phone - right",
    "c3": "texting - left", "c4": "talking on phone - left", "c5": "operating the radio",
    "c6": "drinking", "c7": "reaching behind", "c8": "hair and makeup", "c9": "talking to passenger",
}
print("SMOKE =", SMOKE, "| epochs =", DEEP_EPOCHS)

# %%
# ----------------------------- FIND THE REAL DATA -----------------------------
# Search for driver_imgs_list.csv instead of hard-coding the path (Kaggle: /kaggle/input, Colab: /content/data).
def find_data_root():
    for base in ["/kaggle/input", "/content/data", "data"]:
        for root, dirs, files in os.walk(base):
            if "driver_imgs_list.csv" in files:
                return Path(root)
            dirs[:] = [d for d in dirs if d not in ("test",)]  # don't crawl the 79k unlabelled test images
    raise FileNotFoundError("driver_imgs_list.csv not found (looked in /kaggle/input, /content/data, ./data)")

DATA = find_data_root()
TRAIN_DIR = DATA / "imgs" / "train"
print("Data root:", DATA)

df = pd.read_csv(DATA / "driver_imgs_list.csv")
df["label"] = df["classname"].str[1].astype(int)
df["path"] = [str(TRAIN_DIR / c / f) for c, f in zip(df["classname"], df["img"])]
assert os.path.exists(df["path"].iloc[0]), "image path check failed"
print(f"Full labelled set: {len(df)} images, {df.subject.nunique()} drivers, {df.classname.nunique()} classes")

if SMOKE:
    df = df.sample(n=SMOKE_N_IMAGES, random_state=SEED).reset_index(drop=True)
    print(f"SMOKE: using a random subset of {len(df)} REAL images")
df = df.reset_index(drop=True)
N = len(df)
y_all = df["label"].values

# %%
# ----------------------------- TWO SPLITS (70 / 15 / 15) -----------------------------
from sklearn.model_selection import train_test_split

with Stage("1. Build splits"):
    # (a) RANDOM split: stratified by class, ignores who the driver is.
    idx = np.arange(N)
    r_tr, r_tmp = train_test_split(idx, test_size=0.30, stratify=y_all, random_state=SEED)
    r_va, r_te = train_test_split(r_tmp, test_size=0.50, stratify=y_all[r_tmp], random_state=SEED)

    # (b) DRIVER split: whole drivers go to train OR val OR test, never two of them.
    drivers = sorted(df["subject"].unique())
    rng = np.random.RandomState(SEED)
    rng.shuffle(drivers)
    n_tr = round(0.70 * len(drivers)); n_va = round(0.15 * len(drivers))
    DRIVERS = {"train": sorted(drivers[:n_tr]),
               "val": sorted(drivers[n_tr:n_tr + n_va]),
               "test": sorted(drivers[n_tr + n_va:])}
    d_tr = np.where(df.subject.isin(DRIVERS["train"]))[0]
    d_va = np.where(df.subject.isin(DRIVERS["val"]))[0]
    d_te = np.where(df.subject.isin(DRIVERS["test"]))[0]

    SPLITS = {"random": {"train": r_tr, "val": r_va, "test": r_te},
              "driver": {"train": d_tr, "val": d_va, "test": d_te}}

    # Sanity checks: no image overlap; no driver overlap in the driver split.
    for name, s in SPLITS.items():
        assert len(set(s["train"]) & set(s["test"])) == 0 and len(set(s["train"]) & set(s["val"])) == 0
    assert not (set(DRIVERS["train"]) & set(DRIVERS["test"])) and not (set(DRIVERS["val"]) & set(DRIVERS["test"]))

    split_info = {}
    for name, s in SPLITS.items():
        split_info[name] = {}
        out = df[["img", "subject", "classname"]].copy(); out["split"] = ""
        for part in ["train", "val", "test"]:
            out.loc[s[part], "split"] = part
            subj = sorted(df.loc[s[part], "subject"].unique())
            split_info[name][part] = {"n_images": int(len(s[part])), "fraction": round(len(s[part]) / N, 4),
                                      "n_drivers": len(subj), "drivers": subj}
            print(f"{name:7s} {part:5s}: {len(s[part]):6d} images ({len(s[part])/N:5.1%}), {len(subj):2d} drivers")
        out.to_csv(OUT / "splits" / f"{name}_split.csv", index=False)

    r_tr_drv = set(df.loc[r_tr, "subject"]); r_te_drv = set(df.loc[r_te, "subject"])
    print(f"\nRANDOM split: {len(r_tr_drv & r_te_drv)} of {len(r_te_drv)} test drivers ALSO appear in train  <-- the leak")
    print("DRIVER split drivers:")
    for part in ["train", "val", "test"]:
        print(f"  {part:5s}: {DRIVERS[part]}")

# %%
# ----------------------------- LOAD REAL IMAGES INTO MEMORY (once) -----------------------------
# Each JPEG is decoded once and stored twice: 224x224 (deep model) and 64x64 (classical models).
def load_one(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        big = np.asarray(im.resize((IMG_DEEP, IMG_DEEP), Image.BILINEAR), dtype=np.uint8)
        small = np.asarray(im.resize((IMG_SMALL, IMG_SMALL), Image.BILINEAR), dtype=np.uint8)
    return big, small

with Stage("2. Load images"):
    X224 = np.empty((N, IMG_DEEP, IMG_DEEP, 3), dtype=np.uint8)
    X64 = np.empty((N, IMG_SMALL, IMG_SMALL, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, (big, small) in enumerate(ex.map(load_one, df["path"].values)):
            X224[i] = big; X64[i] = small
            if (i + 1) % 5000 == 0:
                print(f"  loaded {i+1}/{N}", flush=True)
    print(f"X224 {X224.shape} ({X224.nbytes/1e9:.2f} GB), X64 {X64.shape}")

# %%
# ----------------------------- METRICS + PLOTTING HELPERS -----------------------------
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    per_class = cm.diagonal() / np.clip(cm.sum(axis=1), 1, None)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(10)), zero_division=0)),
        "per_class_accuracy": {CLASSES[i]: float(per_class[i]) for i in range(10)},
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_true)),
    }

def per_driver_accuracy(test_idx, y_pred):
    sub = df.loc[test_idx, ["subject"]].copy()
    sub["correct"] = (y_pred == y_all[test_idx])
    g = sub.groupby("subject")["correct"].agg(["mean", "size"])
    return {s: {"accuracy": float(r["mean"]), "n": int(r["size"])} for s, r in g.iterrows()}

def plot_cm(cm, title, path):
    cm = np.array(cm); norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES); ax.set_yticklabels([f"{c} {CLASS_NAMES[c]}" for c in CLASSES], fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=11)
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

RESULTS = {"random": {}, "driver": {}}       # split -> model -> test metrics
VAL_RESULTS = {"random": {}, "driver": {}}   # split -> model -> val accuracy (for reference only)
PER_DRIVER = {}                              # model -> per-test-driver accuracy (driver split)
TRAIN_TIME = {"random": {}, "driver": {}}

def model_slug(name):
    return name.lower().replace(" ", "_")

def record(split, model_name, test_pred, val_pred, train_seconds):
    s = SPLITS[split]
    m = compute_metrics(y_all[s["test"]], test_pred)
    RESULTS[split][model_name] = m
    VAL_RESULTS[split][model_name] = float(accuracy_score(y_all[s["val"]], val_pred))
    TRAIN_TIME[split][model_name] = round(train_seconds, 1)
    if split == "driver":
        PER_DRIVER[model_name] = per_driver_accuracy(s["test"], test_pred)
    plot_cm(m["confusion_matrix"], f"{model_name} — {split} split (test acc {m['accuracy']:.1%})",
            FIG / f"cm_{split}_{model_slug(model_name)}.png")
    print(f"  {model_name:14s} [{split}]  test acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  "
          f"(val acc {VAL_RESULTS[split][model_name]:.4f}, train {train_seconds:.0f}s)", flush=True)

# %%
# ----------------------------- CLASSICAL BASELINES (like CS229) -----------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def flat(idx):
    return X64[idx].reshape(len(idx), -1).astype(np.float32) / 255.0

def classical_models():
    return {
        "Naive Bayes": GaussianNB(),
        "Linear SVM": SGDClassifier(loss="hinge", alpha=1e-4, max_iter=50, tol=1e-3, random_state=SEED, n_jobs=-1),
        "Softmax": LogisticRegression(max_iter=300, random_state=SEED),  # multinomial logistic regression
        "Decision Tree": DecisionTreeClassifier(max_depth=TREE_MAX_DEPTH, random_state=SEED),
    }

CLASSICAL_TRAIN_N = {}
for split in ["random", "driver"]:
    with Stage(f"3. Classical models — {split} split"):
        tr = SPLITS[split]["train"]
        if len(tr) > CLASSICAL_MAX_TRAIN:   # stratified subsample of the TRAIN set only
            tr, _ = train_test_split(tr, train_size=CLASSICAL_MAX_TRAIN, stratify=y_all[tr], random_state=SEED)
        CLASSICAL_TRAIN_N[split] = int(len(tr))
        print(f"  training on {len(tr)} of {len(SPLITS[split]['train'])} train images")
        scaler = StandardScaler().fit(flat(tr))          # fitted on TRAIN only
        Xtr = scaler.transform(flat(tr)); ytr = y_all[tr]
        Xva = scaler.transform(flat(SPLITS[split]["val"]))
        Xte = scaler.transform(flat(SPLITS[split]["test"]))
        for name, model in classical_models().items():
            t = time.time()
            model.fit(Xtr, ytr)
            record(split, name, model.predict(Xte), model.predict(Xva), time.time() - t)
        del Xtr, Xva, Xte

# %%
# ----------------------------- DEEP MODEL: MobileNetV3-Large -----------------------------
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models

assert torch.cuda.is_available(), "No GPU found: turn on a GPU accelerator"
DEVICE = "cuda"
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
torch.backends.cudnn.benchmark = True

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

def batches(idx, batch_size, shuffle=False, drop_last=False):
    """Yield (uint8 images on GPU, labels on GPU) straight from the in-memory array: no worker processes."""
    idx = np.asarray(idx)
    if shuffle:
        idx = idx[np.random.permutation(len(idx))]
    stop = len(idx) - (len(idx) % batch_size if drop_last else 0)
    for i in range(0, stop, batch_size):
        b = np.sort(idx[i:i + batch_size])            # sorted reads are faster on a big array
        x = torch.from_numpy(X224[b]).pin_memory().to(DEVICE, non_blocking=True).permute(0, 3, 1, 2)
        y = torch.from_numpy(y_all[b]).to(DEVICE, non_blocking=True)
        yield x, y

def gpu_augment(x):
    """Light augmentation for a whole batch on the GPU (x: float in [0,1], B x 3 x H x W).
    Small random rotation (+-8 deg), shift (+-5%), zoom (0.9-1.1) and colour jitter.
    NO horizontal flip: "texting - right" vs "texting - left" would get mixed up."""
    B = x.size(0); dev = x.device
    ang = (torch.rand(B, device=dev) * 2 - 1) * (8 * np.pi / 180)
    scale = 0.9 + 0.2 * torch.rand(B, device=dev)
    tx, ty = [(torch.rand(B, device=dev) * 2 - 1) * 0.1 for _ in range(2)]   # 0.1 in [-1,1] coords = 5% of the image
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    b = 0.7 + 0.6 * torch.rand(B, 1, 1, 1, device=dev)                      # brightness 0.7-1.3
    c = 0.7 + 0.6 * torch.rand(B, 1, 1, 1, device=dev)                      # contrast 0.7-1.3
    sat = 0.8 + 0.4 * torch.rand(B, 1, 1, 1, device=dev)                    # saturation 0.8-1.2
    x = x * b
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = (x - gray.mean(dim=(2, 3), keepdim=True)) * c + gray.mean(dim=(2, 3), keepdim=True)
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = (x - gray) * sat + gray
    return x.clamp(0, 1)

def to_input(x_uint8, augment=False):
    """uint8 batch on GPU -> (optionally augmented) normalized float batch (ImageNet mean/std)."""
    x = x_uint8.float() / 255.0
    if augment:
        x = gpu_augment(x)
    return (x - MEAN) / STD

def make_model(pretrained=True):
    w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.mobilenet_v3_large(weights=w)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 10)
    return m

@torch.no_grad()
def predict(model, idx):
    model.train(False)   # evaluation mode (no dropout, fixed batch-norm)
    order = np.asarray(idx)
    probs = np.empty((len(order), 10), dtype=np.float32)
    for i in range(0, len(order), 256):
        b = order[i:i + 256]
        x = torch.from_numpy(X224[b]).to(DEVICE).permute(0, 3, 1, 2)
        with torch.autocast("cuda", dtype=torch.float16):
            probs[i:i + len(b)] = F.softmax(model(to_input(x)).float(), 1).cpu().numpy()
    return probs

def train_deep(split):
    set_seed()
    s = SPLITS[split]
    model = make_model().to(DEVICE).to(memory_format=torch.channels_last)
    steps_per_epoch = len(s["train"]) // BATCH_SIZE
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=DEEP_EPOCHS * steps_per_epoch, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    history, best_acc, best_state, best_epoch = [], -1.0, None, -1
    for epoch in range(1, DEEP_EPOCHS + 1):
        model.train(); t = time.time(); tot, n = torch.zeros((), device=DEVICE), 0
        for x, y in batches(s["train"], BATCH_SIZE, shuffle=True, drop_last=True):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(to_input(x, augment=True).contiguous(memory_format=torch.channels_last)), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.detach() * len(y); n += len(y)
        torch.cuda.synchronize(); t_train = time.time() - t
        tot = float(tot)
        val_acc = float((predict(model, s["val"]).argmax(1) == y_all[s["val"]]).mean())
        history.append({"epoch": epoch, "train_loss": tot / n, "val_acc": val_acc, "seconds": round(time.time() - t, 1),
                        "train_images_per_sec": round(n / t_train, 1)})
        flag = ""
        if val_acc > best_acc:   # model selection uses VALIDATION only
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  <- best so far"
        print(f"  epoch {epoch}/{DEEP_EPOCHS}  loss {tot/n:.4f}  val acc {val_acc:.4f}  ({time.time()-t:.0f}s, {n/t_train:.0f} img/s){flag}", flush=True)
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "arch": "mobilenet_v3_large", "classes": CLASSES,
                "class_names": CLASS_NAMES, "img_size": IMG_DEEP,
                "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                "split": split, "best_epoch": best_epoch, "best_val_acc": best_acc,
                "smoke": SMOKE}, OUT / f"model_{split}.pt")
    return model, history, best_epoch

DEEP = "MobileNetV3"
DEEP_MODELS, DEEP_HISTORY, DEEP_BEST_EPOCH = {}, {}, {}
for split in ["random", "driver"]:
    with Stage(f"4. MobileNetV3 — {split} split"):
        t = time.time()
        model, hist, best_ep = train_deep(split)
        train_s = time.time() - t
        DEEP_MODELS[split], DEEP_HISTORY[split], DEEP_BEST_EPOCH[split] = model, hist, best_ep
        test_pred = predict(model, SPLITS[split]["test"]).argmax(1)
        val_pred = predict(model, SPLITS[split]["val"]).argmax(1)
        record(split, DEEP, test_pred, val_pred, train_s)

# %%
# ----------------------------- PICK SHARED TEST IMAGES (Grad-CAM + demo) -----------------------------
# We need images that NEITHER model trained on: in the random-split test set AND from a driver-split test driver.
with Stage("5. Grad-CAM + sample images"):
    shared = np.array(sorted(set(SPLITS["random"]["test"]) & set(SPLITS["driver"]["test"])))
    print(f"  {len(shared)} images are unseen by BOTH models")
    rng = np.random.RandomState(SEED)
    samples = []
    for c in range(10):
        pool = shared[y_all[shared] == c]
        samples += list(rng.choice(pool, size=min(N_SAMPLES_PER_CLASS, len(pool)), replace=False))
    samples = np.array(samples, dtype=int)
    # Grad-CAM set: one per class first, then fill up to N_GRADCAM.
    firsts = [int(samples[y_all[samples] == c][0]) for c in range(10) if (y_all[samples] == c).any()]
    rest = [int(i) for i in samples if int(i) not in firsts]
    gradcam_idx = np.array((firsts + rest)[:N_GRADCAM], dtype=int)

    # Copy the ORIGINAL real JPEGs for the demo gallery.
    manifest = []
    for i in samples:
        r = df.loc[i]
        fname = f"{r.classname}_{r.subject}_{r.img}"
        shutil.copy(r.path, OUT / "sample_images" / fname)
        manifest.append({"file": fname, "img": r.img, "subject": r.subject, "true_class": r.classname,
                         "true_name": CLASS_NAMES[r.classname]})
    json.dump(manifest, open(OUT / "sample_images" / "manifest.json", "w"), indent=1)

    def grad_cam(model, i):
        """Grad-CAM on the last conv block (features[-1]) for the predicted class."""
        model.train(False); store = {}
        def fwd_hook(m, inp, out):
            store["act"] = out
            out.register_hook(lambda g: store.__setitem__("grad", g))
        h = model.features[-1].register_forward_hook(fwd_hook)
        x = to_input(torch.from_numpy(X224[i]).permute(2, 0, 1)[None].to(DEVICE))
        with torch.enable_grad():
            logits = model(x)
            pred = int(logits.argmax(1))
            model.zero_grad(); logits[0, pred].backward()
        h.remove()
        w = store["grad"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * store["act"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_DEEP, IMG_DEEP), mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        prob = float(F.softmax(logits, 1)[0, pred])
        return cam.detach().cpu().numpy(), pred, prob

    gc_records = []
    ncols = 2  # two images per row, each shown as [image | random model | driver model]
    nrows = max(1, int(np.ceil(len(gradcam_idx) / ncols)))
    fig, axes = plt.subplots(nrows, 3 * ncols, figsize=(3 * ncols * 2.4, nrows * 2.5))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel(): ax.axis("off")
    for k, i in enumerate(gradcam_idx):
        r, c0 = divmod(k, ncols); c0 *= 3
        rec = {"img": df.loc[i, "img"], "subject": df.loc[i, "subject"], "true": CLASSES[y_all[i]]}
        axes[r, c0].imshow(X224[i]); axes[r, c0].set_title(f"true {CLASSES[y_all[i]]}: {CLASS_NAMES[CLASSES[y_all[i]]]}", fontsize=7)
        for off, split in [(1, "random"), (2, "driver")]:
            cam, pred, prob = grad_cam(DEEP_MODELS[split], i)
            rec[f"pred_{split}"] = CLASSES[pred]; rec[f"prob_{split}"] = prob
            ax = axes[r, c0 + off]
            ax.imshow(X224[i]); ax.imshow(cam, cmap="jet", alpha=0.45)
            ok = "correct" if pred == y_all[i] else "wrong"
            ax.set_title(f"{split} model: {CLASSES[pred]} {prob:.0%} ({ok})", fontsize=7,
                         color="green" if pred == y_all[i] else "red")
        gc_records.append(rec)
    fig.suptitle("Grad-CAM: where each MobileNetV3 looks (same unseen test images)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "gradcam_grid.png", dpi=150); plt.close(fig)

# %%
# ----------------------------- SUMMARY FIGURES + metrics.json -----------------------------
with Stage("6. Figures + metrics.json"):
    model_order = ["Naive Bayes", "Linear SVM", "Softmax", "Decision Tree", DEEP]

    # Headline chart: random vs driver split accuracy for every model.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    xs = np.arange(len(model_order)); w = 0.38
    ra = [RESULTS["random"][m]["accuracy"] for m in model_order]
    da = [RESULTS["driver"][m]["accuracy"] for m in model_order]
    b1 = ax.bar(xs - w/2, ra, w, label="Random split (seen drivers)", color="#4C72B0")
    b2 = ax.bar(xs + w/2, da, w, label="Driver split (new drivers)", color="#DD8452")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{b.get_height():.0%}", ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(model_order); ax.set_ylim(0, 1.08); ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy drops when test drivers are new"); ax.legend(loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "split_comparison.png", dpi=150); plt.close(fig)

    # Per-test-driver accuracy (driver split).
    test_drivers = DRIVERS["test"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.8 / len(model_order)
    for k, m in enumerate(model_order):
        vals = [PER_DRIVER[m].get(d, {"accuracy": np.nan})["accuracy"] for d in test_drivers]
        ax.bar(np.arange(len(test_drivers)) + (k - len(model_order)/2 + 0.5) * w, vals, w, label=m)
    ax.set_xticks(range(len(test_drivers))); ax.set_xticklabels(test_drivers)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.05); ax.set_title("Driver split: accuracy for each unseen test driver")
    ax.legend(fontsize=8, ncol=3); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "per_driver_accuracy.png", dpi=150); plt.close(fig)

    # Training curves.
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for split, col in [("random", "#4C72B0"), ("driver", "#DD8452")]:
        h = DEEP_HISTORY[split]
        ax.plot([e["epoch"] for e in h], [e["val_acc"] for e in h], "o-", color=col, label=f"{split} split — val acc")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation accuracy"); ax.set_ylim(0, 1.02)
    ax.set_title("MobileNetV3 validation accuracy per epoch"); ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "training_curves.png", dpi=150); plt.close(fig)

    # One real example per class (for the dataset slide).
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.6))
    for c, ax in enumerate(axes.ravel()):
        i = np.where(y_all == c)[0][0]
        ax.imshow(X224[i]); ax.axis("off"); ax.set_title(f"{CLASSES[c]}: {CLASS_NAMES[CLASSES[c]]}", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "class_examples.png", dpi=130); plt.close(fig)

    # Class balance.
    counts = df["classname"].value_counts().reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(CLASSES, counts.values, color="#4C72B0"); ax.set_ylabel("Images"); ax.set_title("Images per class")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "class_counts.png", dpi=150); plt.close(fig)

    TIMINGS["total"] = round(time.time() - T_START, 1)
    metrics = {
        "meta": {
            "project": "Machine Learning Techniques for Distracted Driver Detection",
            "team": "F-28", "smoke": SMOKE, "seed": SEED,
            "n_images_used": int(N), "n_drivers": int(df.subject.nunique()),
            "deep_model": "MobileNetV3-Large (ImageNet-pretrained, torchvision IMAGENET1K_V2)",
            "deep_epochs": DEEP_EPOCHS, "deep_batch_size": BATCH_SIZE, "deep_lr": LR, "deep_img_size": IMG_DEEP,
            "deep_best_epoch": DEEP_BEST_EPOCH,
            "classical_img_size": IMG_SMALL, "classical_max_train": CLASSICAL_MAX_TRAIN, "tree_max_depth": TREE_MAX_DEPTH,
            "augmentation": "on GPU: rotation +-8 deg, shift +-5%, zoom 0.9-1.1, brightness/contrast 0.7-1.3, saturation 0.8-1.2, no flip",
            "classical_train_n": CLASSICAL_TRAIN_N,
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "timings_seconds": TIMINGS,
        },
        "classes": CLASSES, "class_names": CLASS_NAMES,
        "class_counts": {c: int(v) for c, v in counts.items()},
        "splits": split_info,
        "driver_split_drivers": DRIVERS,
        "models": model_order,
        "test": RESULTS,
        "val_accuracy": VAL_RESULTS,
        "train_seconds": TRAIN_TIME,
        "drop": {m: {"random_acc": RESULTS["random"][m]["accuracy"], "driver_acc": RESULTS["driver"][m]["accuracy"],
                     "drop_points": 100 * (RESULTS["random"][m]["accuracy"] - RESULTS["driver"][m]["accuracy"])}
                 for m in model_order},
        "per_test_driver_accuracy": PER_DRIVER,
        "deep_history": DEEP_HISTORY,
        "gradcam": gc_records,
        "sample_images": manifest,
    }
    json.dump(metrics, open(OUT / "metrics.json", "w"), indent=1)

# %%
print("\n================ SUMMARY ================")
print(f"{'model':16s} {'random acc':>11s} {'driver acc':>11s} {'drop (pts)':>11s}")
for m in model_order:
    d = metrics["drop"][m]
    print(f"{m:16s} {d['random_acc']:11.4f} {d['driver_acc']:11.4f} {d['drop_points']:11.1f}")
print("\nTimings (s):", json.dumps(TIMINGS, indent=1))
print("\nOutput files:")
for p in sorted(OUT.rglob("*")):
    if p.is_file() and "sample_images" not in str(p):
        print(f"  {p.relative_to(OUT)}  ({p.stat().st_size/1e6:.2f} MB)")
print(f"  sample_images/: {len(list((OUT/'sample_images').glob('*.jpg')))} jpgs")
