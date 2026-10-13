"""Driver-grouped cross-validation of the baseline and the two best debiasing recipes, with harder identity probes.

Runs ON a Colab GPU (sent by scripts/colab_cv.sh, one seed per job). Uses the REAL State Farm data.
Folds: the 26 drivers are shuffled once (seed 42) and cut into 5 groups. For fold k, group k = TEST drivers (unseen),
group k+1 = VALIDATION drivers (chooses the epoch), the other 3 groups = TRAINING drivers. Fold assignment is the same for every seed.
Variants (MobileNetV3-Large, ImageNet-pretrained, 8 epochs, AdamW + one-cycle, label smoothing 0.1):
  baseline : light augmentation (as the main run)
  A        : appearance-scrambling augmentation (random erasing, random grayscale, strong colour jitter)
  C        : A + adversarial driver head behind a gradient-reversal layer (lambda ramps to 1.0)
Identity probes on each fold's TEST drivers (frozen, L2-normalised 960-d pooled features; analysis only, never used for selection):
  one_shot_id : within one action class, give one photo per driver as reference; match every other photo of that class to the
                nearest reference (cosine). Accuracy, averaged over 20 random draws and all classes. Chance = 1 / number of test drivers.
  verif_auc   : within one action class, is cosine similarity higher for same-driver pairs than different-driver pairs? ROC AUC
                over 4000 random pairs per class (0.5 = identity cannot be told apart).
"""
import json, math, os, random, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

SEEDS = [0]                      # replaced per job by scripts/colab_cv.sh
EPOCHS, BS, LR, IMG, N_FOLDS = 8, 64, 1e-3, 224, 5
VARIANTS = [("baseline", False, 0.0), ("A", True, 0.0), ("C", True, 1.0)]
T0 = time.time()
DEVICE = "cuda"; assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
print("GPU:", torch.cuda.get_device_name(0), "| seeds", SEEDS, flush=True)
OUT = Path("/content/cv"); OUT.mkdir(parents=True, exist_ok=True)
CLASSES = [f"c{i}" for i in range(10)]

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

DATA = Path("/content/data")
df = pd.read_csv(DATA / "driver_imgs_list.csv").reset_index(drop=True)
df["label"] = df["classname"].str[1].astype(int)
y_all = df["label"].values; subj_all = df["subject"].values
drivers = np.array(sorted(df["subject"].unique()))
groups = np.array_split(drivers[np.random.RandomState(42).permutation(len(drivers))], N_FOLDS)
FOLDS = []
for k in range(N_FOLDS):
    te, va = list(groups[k]), list(groups[(k + 1) % N_FOLDS])
    tr = [d for d in drivers if d not in te and d not in va]
    FOLDS.append({"train": tr, "val": va, "test": te})
    print(f"fold {k}: test {te} | val {va} | {len(tr)} train drivers", flush=True)

t = time.time()
def load_one(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB").resize((IMG, IMG), Image.BILINEAR), dtype=np.uint8)
paths = [str(DATA / "imgs" / "train" / c / f) for c, f in zip(df["classname"], df["img"])]
X = np.empty((len(df), IMG, IMG, 3), dtype=np.uint8)
with ThreadPoolExecutor(16) as ex:
    for i, a in enumerate(ex.map(load_one, paths)):
        X[i] = a
print(f"loaded {len(df)} images in {(time.time()-t)/60:.1f} min", flush=True)

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

def gray(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

def augment(x, strong):
    """Batch augmentation on the GPU. Base = the main run's light augmentation (no horizontal flip).
    strong=True adds appearance scrambling: wider colour jitter, random grayscale (p=0.2) and random erasing (p=0.5)."""
    B, dev = x.size(0), x.device
    u = lambda lo, hi, *shape: lo + (hi - lo) * torch.rand(*(shape or (B,)), device=dev)
    ang, scale = u(-8, 8) * math.pi / 180, u(0.9, 1.1)
    tx, ty = u(-0.1, 0.1), u(-0.1, 0.1)
    c, s = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([c, -s, tx], 1), torch.stack([s, c, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False), align_corners=False)
    lo_b, hi_b, lo_s, hi_s = (0.6, 1.4, 0.5, 1.5) if strong else (0.7, 1.3, 0.8, 1.2)
    x = x * u(lo_b, hi_b, B, 1, 1, 1)
    m = gray(x).mean(dim=(2, 3), keepdim=True)
    x = (x - m) * u(lo_b, hi_b, B, 1, 1, 1) + m
    g = gray(x); x = (x - g) * u(lo_s, hi_s, B, 1, 1, 1) + g
    if strong:
        sel = (torch.rand(B, 1, 1, 1, device=dev) < 0.2)
        x = torch.where(sel, gray(x).expand_as(x), x)
        H, W = x.shape[2:]
        area = u(0.02, 0.2) * H * W
        ratio = torch.exp(u(math.log(0.3), math.log(3.3)))
        h = torch.sqrt(area * ratio).clamp(max=H); w = torch.sqrt(area / ratio).clamp(max=W)
        y0 = torch.rand(B, device=dev) * (H - h); x0 = torch.rand(B, device=dev) * (W - w)
        ys = torch.arange(H, device=dev).view(1, H, 1); xs = torch.arange(W, device=dev).view(1, 1, W)
        box = (ys >= y0.view(B, 1, 1)) & (ys < (y0 + h).view(B, 1, 1)) & (xs >= x0.view(B, 1, 1)) & (xs < (x0 + w).view(B, 1, 1))
        box = box & (torch.rand(B, device=dev) < 0.5).view(B, 1, 1)
        x = torch.where(box[:, None], torch.rand_like(x), x)
    return x.clamp(0, 1)

def to_input(xu8, aug=None):
    x = xu8.float() / 255.0
    if aug is not None:
        x = augment(x, strong=aug)
    return ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)

def gpu_batch(b):
    return torch.from_numpy(X[b]).pin_memory().to(DEVICE, non_blocking=True).permute(0, 3, 1, 2)

# ---------------- model with optional adversarial driver head ----------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None

class Net(nn.Module):
    """MobileNetV3-Large; the driver head sees the 960-d pooled features through a gradient-reversal layer,
    so the backbone is pushed to make those features useless for telling drivers apart."""
    def __init__(self, n_drivers=0):
        super().__init__()
        self.m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        self.m.classifier[3] = nn.Linear(1280, 10)
        self.drv = nn.Sequential(nn.Linear(960, 256), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(256, n_drivers)) if n_drivers else None
    def feats(self, x):
        return self.m.avgpool(self.m.features(x)).flatten(1)
    def forward(self, x, lam=0.0):
        f = self.feats(x)
        return self.m.classifier(f), (self.drv(GradReverse.apply(f, lam)) if self.drv is not None else None)


@torch.no_grad()
def predict(m, ids):
    m.train(False); out = np.empty((len(ids), 10), dtype=np.float32)
    for i in range(0, len(ids), 256):
        b = ids[i:i + 256]
        with torch.autocast("cuda", dtype=torch.float16):
            out[i:i + len(b)] = F.softmax(m(to_input(gpu_batch(b))).float(), 1).cpu().numpy()
    return out

@torch.no_grad()
def features(m, ids):
    m.train(False); fs = []
    for i in range(0, len(ids), 256):
        with torch.autocast("cuda", dtype=torch.float16):
            fs.append(m.avgpool(m.features(to_input(gpu_batch(ids[i:i + 256])))).flatten(1).float().cpu().numpy())
    f = np.concatenate(fs)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)

def identity_probes(feat, ids, seed):
    rng = np.random.RandomState(1000 + seed); subj = subj_all[ids]; lab = y_all[ids]
    test_d = sorted(set(subj)); accs, aucs = [], []
    for c in range(10):
        in_c = np.where(lab == c)[0]
        by_d = {d: in_c[subj[in_c] == d] for d in test_d}
        if min(len(v) for v in by_d.values()) < 2:
            continue
        for _ in range(20):                                   # one-shot identification
            refs = np.array([rng.choice(by_d[d]) for d in test_d])
            q = np.setdiff1d(in_c, refs)
            pred = np.array(test_d)[np.argmax(feat[q] @ feat[refs].T, axis=1)]
            accs.append(float((pred == subj[q]).mean()))
        a, b = rng.choice(in_c, 4000), rng.choice(in_c, 4000)   # verification pairs
        keep = a != b
        same = (subj[a] == subj[b])[keep]; sim = np.sum(feat[a[keep]] * feat[b[keep]], 1)
        if same.any() and (~same).any():
            aucs.append(float(roc_auc_score(same, sim)))
    return {"one_shot_id": float(np.mean(accs)), "verif_auc": float(np.mean(aucs)), "chance_one_shot": 1 / len(test_d)}

def run(fold_k, seed, name, strong, lam_max):
    fold = FOLDS[fold_k]; seed_all(seed); t = time.time()
    ix = {p: np.where(np.isin(subj_all, fold[p]))[0] for p in ["train", "val", "test"]}
    drv_id = {d: i for i, d in enumerate(fold["train"])}
    d_all = np.array([drv_id.get(s, -1) for s in subj_all])
    net = Net(len(fold["train"]) if lam_max > 0 else 0).to(DEVICE).to(memory_format=torch.channels_last)
    steps = EPOCHS * (len(ix["train"]) // BS)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda"); ce = nn.CrossEntropyLoss(label_smoothing=0.1); ce_d = nn.CrossEntropyLoss()
    best, best_state, best_ep, step, hist = -1, None, -1, 0, []
    for ep in range(1, EPOCHS + 1):
        net.train(); order = ix["train"][np.random.permutation(len(ix["train"]))]
        for i in range(0, len(order) - BS + 1, BS):
            b = np.sort(order[i:i + BS]); x, y = gpu_batch(b), torch.from_numpy(y_all[b]).to(DEVICE)
            lam = lam_max * (2 / (1 + math.exp(-10 * step / steps)) - 1)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, dlog = net(to_input(x, aug=strong), lam)
                loss = ce(logits, y)
                if dlog is not None:
                    loss = loss + ce_d(dlog, torch.from_numpy(d_all[b]).to(DEVICE))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            step += 1
        va = float((predict(net.m, ix["val"]).argmax(1) == y_all[ix["val"]]).mean()); hist.append(va)
        if va > best:
            best, best_ep = va, ep
            best_state = {k: v.detach().cpu().clone() for k, v in net.m.state_dict().items()}
    net.m.load_state_dict(best_state)
    pred = predict(net.m, ix["test"]).argmax(1); yt = y_all[ix["test"]]
    cm = confusion_matrix(yt, pred, labels=list(range(10)))
    rec = {"fold": fold_k, "seed": seed, "variant": name, "test_drivers": fold["test"], "val_drivers": fold["val"],
           "val_acc": best, "best_epoch": best_ep, "val_curve": hist,
           "test_acc": float((pred == yt).mean()), "test_macro_f1": float(f1_score(yt, pred, average="macro")),
           "per_class_acc": {CLASSES[i]: float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(10)},
           "per_driver_acc": {d: float((pred[subj_all[ix["test"]] == d] == yt[subj_all[ix["test"]] == d]).mean()) for d in fold["test"]},
           **identity_probes(features(net.m, ix["test"]), ix["test"], seed), "minutes": round((time.time() - t) / 60, 2)}
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  fold {fold_k} seed {seed} {name:8s}: val {best:.4f} (ep {best_ep}) test {rec['test_acc']:.4f} F1 {rec['test_macro_f1']:.4f} "
          f"| 1-shot id {rec['one_shot_id']:.3f} (chance {rec['chance_one_shot']:.2f}) verif AUC {rec['verif_auc']:.3f} | {rec['minutes']} min", flush=True)
    del net; torch.cuda.empty_cache()

# SETUP_ONLY=True (set by scripts/colab_cv.sh) loads data and defines run(); training is then driven fold by fold.
if not globals().get("SETUP_ONLY", False):
    # ImageNet-only reference probes (no fine-tuning), once per fold
    ref = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2).to(DEVICE).to(memory_format=torch.channels_last)
    for k, fold in enumerate(FOLDS):
        ids = np.where(np.isin(subj_all, fold["test"]))[0]
        rec = {"fold": k, "seed": SEEDS[0], "variant": "imagenet", **identity_probes(features(ref, ids), ids, SEEDS[0])}
        with open(OUT / "results.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  fold {k} imagenet-only: 1-shot id {rec['one_shot_id']:.3f} verif AUC {rec['verif_auc']:.3f}", flush=True)
    del ref

    for seed in SEEDS:
        for k in range(N_FOLDS):
            for name, strong, lam in VARIANTS:
                run(k, seed, name, strong, lam)
    print(f"CV JOB DONE in {(time.time()-T0)/60:.1f} min", flush=True)
