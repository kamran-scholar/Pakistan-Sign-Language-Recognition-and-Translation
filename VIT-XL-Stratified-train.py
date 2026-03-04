# train_vit.py
# Usage example:
#   conda activate vit-env
#   python train_vit.py --train-dir "/path/to/train" --val-dir "/path/to/val" --out-dir "./out"
#
# The code below is adapted from the snippet you provided. Paths are now configurable via CLI flags.
# All original logic is preserved.

import os, random, argparse
from contextlib import contextmanager

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm
from timm.loss import SoftTargetCrossEntropy
from timm.data import Mixup, resolve_model_data_config
from timm.scheduler import CosineLRScheduler
from timm.utils import ModelEmaV2

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

TRAIN_DIR = "/home/gpu/PSL Isolated Signs Datasets/splits-of-combined-cropped-dataset-18-8/train"
VAL_DIR   = "/home/gpu/PSL Isolated Signs Datasets/splits-of-combined-cropped-dataset-18-8/val"
OUT_DIR   = "/home/gpu/PSL Isolated Signs Datasets/ConvNext-XL-Stratified-checkpoint"




# ==================== DEFAULT CONFIG (overridable via CLI) ====================
DEFAULTS = dict(
    train_dir="/home/gpu/PSL Isolated Signs Datasets/splits-of-combined-cropped-dataset-18-8/train",
    val_dir="/home/gpu/PSL Isolated Signs Datasets/splits-of-combined-cropped-dataset-18-8/val",
    out_dir="/home/gpu/PSL Isolated Signs Datasets/VIT-XL-Stratified-Train-CHKPT",
    model_name="deit3_large_patch16_224.fb_in22k_ft_in1k",
    num_classes=39,
    epochs=100,
    freeze_epochs=3,
    batch_size=50,
    accum_steps=2,
    label_smooth=0.1,
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    mixup_prob=1.0,
    mixup_switch_prob=0.5,
    drop_rate=0.0,
    drop_path_rate=0.25,
    base_lr_full=5e-5,
    base_lr_linear=1e-3,
    weight_decay=0.05,
    warmup_epochs=5,
    grad_clip_norm=1.0,
    early_stop_patience=15,
    random_seed=42
)

torch.backends.cudnn.benchmark = True

# ==================== AMP SHIM (torch 1.x / 2.x) ====================
USE_TORCH_AMP = hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
if USE_TORCH_AMP:
    def autocast_ctx():
        return torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available())
    GradScaler = torch.amp.GradScaler
else:
    from torch.cuda.amp import autocast as _autocast_cuda
    from torch.cuda.amp import GradScaler as _GradScaler
    def autocast_ctx():
        return _autocast_cuda(enabled=torch.cuda.is_available())
    GradScaler = _GradScaler

# ==================== HARD DISABLE CHECKPOINTING (Torch + timm) ====================
def _make_no_ckpt_fn():
    def _no_ckpt(function, *args, **kwargs):
        return function(*args)
    return _no_ckpt

def _make_no_ckpt_seq_fn():
    def _no_ckpt_seq(functions, *args, **kwargs):
        x = args[0] if len(args) > 0 else None
        for fn in functions:
            x = fn(x)
        return x
    return _no_ckpt_seq

@contextmanager
def disable_all_checkpointing():
    patches = []

    try:
        import torch.utils.checkpoint as _cp
        if hasattr(_cp, "checkpoint"):
            patches.append((_cp, "checkpoint", _cp.checkpoint, _make_no_ckpt_fn()))
    except Exception:
        pass

    try:
        import timm.models.layers as tml
        if hasattr(tml, "checkpoint"):
            patches.append((tml, "checkpoint", tml.checkpoint, _make_no_ckpt_fn()))
        if hasattr(tml, "checkpoint_seq"):
            patches.append((tml, "checkpoint_seq", tml.checkpoint_seq, _make_no_ckpt_seq_fn()))
    except Exception:
        pass

    try:
        import timm.layers as tl
        if hasattr(tl, "checkpoint"):
            patches.append((tl, "checkpoint", tl.checkpoint, _make_no_ckpt_fn()))
        if hasattr(tl, "checkpoint_seq"):
            patches.append((tl, "checkpoint_seq", tl.checkpoint_seq, _make_no_ckpt_seq_fn()))
    except Exception:
        pass

    for mod, name, orig, repl in patches:
        try:
            setattr(mod, name, repl if callable(repl) else repl())
        except Exception:
            pass

    try:
        yield
    finally:
        for mod, name, orig, repl in patches[::-1]:
            try:
                setattr(mod, name, orig)
            except Exception:
                pass

# ==================== UTILS ====================
def set_seed(seed=DEFAULTS["random_seed"]):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def safe_imread_rgb(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

class AlbumentationsImageFolder(ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root=root); self.atransform = transform
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = safe_imread_rgb(path)
        img = self.atransform(image=img)["image"]
        return img, target

def make_train_tfms(img_size, mean, std):
    return A.Compose([
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.55, 1.0), ratio=(0.8, 1.25)),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=tuple(mean), std=tuple(std)),
        ToTensorV2(),
    ])

def make_val_tfms(img_size, mean, std):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_REFLECT_101),
        A.Normalize(mean=tuple(mean), std=tuple(std)),
        ToTensorV2(),
    ])

def make_loader(ds, batch_size, shuffle, workers=8):
    return DataLoader(
        dataset=ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=True, drop_last=shuffle, persistent_workers=True
    )

def build_model(num_classes, model_name, drop_rate, drop_path_rate):
    model = timm.create_model(
        model_name=model_name,
        pretrained=True,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    )
    if hasattr(model, "set_grad_checkpointing"):
        try: model.set_grad_checkpointing(enable=False)
        except TypeError: pass
    if hasattr(model, "grad_checkpointing"):
        try: model.grad_checkpointing = False
        except Exception: pass
    return model

def set_backbone_trainable(m: nn.Module, trainable: bool):
    head_names = ['head', 'fc', 'classifier', 'heads']
    for n, p in m.named_parameters():
        p.requires_grad = trainable or any(h in n for h in head_names)

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    crit = nn.CrossEntropyLoss()
    y_true, y_pred = [], []
    loss_meter, total = 0.0, 0
    top5_correct = 0
    k = min(5, num_classes)

    for images, targets in loader:
        assert images.ndim == 4, "Images must be NCHW [B,C,H,W]"
        assert targets.ndim == 1, "Targets must be 1D class indices"

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with disable_all_checkpointing():
            with autocast_ctx():
                logits = model(images)
                loss = crit(logits, targets)

        assert logits.ndim == 2 and logits.shape[1] == num_classes, \
            f"Logits must be [B, {num_classes}] but got {tuple(logits.shape)}"
        assert targets.max().item() < num_classes, "Target index exceeds num_classes"

        loss_meter += loss.item() * images.size(0)
        total += images.size(0)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        y_true.extend(targets.detach().cpu().numpy())
        y_pred .extend(preds.detach().cpu().numpy())

        topk_idx = torch.topk(probs, k=k, dim=1).indices
        top5_correct += topk_idx.eq(targets.view(-1, 1)).any(dim=1).sum().item()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    top1 = acc * 100.0
    top5 = 100.0 * (top5_correct / max(1, total))

    return {
        "acc": acc * 100.0,
        "prec": prec * 100.0,
        "rec": rec * 100.0,
        "f1": f1 * 100.0,
        "cm": cm,
        "loss": loss_meter / max(1, total),
        "top1": top1,
        "top5": top5
    }

def parse_args():
    p = argparse.ArgumentParser(description="Train tiny ViT/DeiT model with timm + albumentations")
    p.add_argument("--train-dir", default=DEFAULTS["train_dir"], type=str)
    p.add_argument("--val-dir",   default=DEFAULTS["val_dir"],   type=str)
    p.add_argument("--out-dir",   default=DEFAULTS["out_dir"],   type=str)

    p.add_argument("--model-name", default=DEFAULTS["model_name"], type=str)
    p.add_argument("--num-classes", default=DEFAULTS["num_classes"], type=int)

    p.add_argument("--epochs", default=DEFAULTS["epochs"], type=int)
    p.add_argument("--freeze-epochs", default=DEFAULTS["freeze_epochs"], type=int)

    p.add_argument("--batch-size", default=DEFAULTS["batch_size"], type=int)
    p.add_argument("--accum-steps", default=DEFAULTS["accum_steps"], type=int)

    p.add_argument("--label-smooth", default=DEFAULTS["label_smooth"], type=float)
    p.add_argument("--mixup-alpha", default=DEFAULTS["mixup_alpha"], type=float)
    p.add_argument("--cutmix-alpha", default=DEFAULTS["cutmix_alpha"], type=float)
    p.add_argument("--mixup-prob", default=DEFAULTS["mixup_prob"], type=float)
    p.add_argument("--mixup-switch-prob", default=DEFAULTS["mixup_switch_prob"], type=float)

    p.add_argument("--drop-rate", default=DEFAULTS["drop_rate"], type=float)
    p.add_argument("--drop-path-rate", default=DEFAULTS["drop_path_rate"], type=float)

    p.add_argument("--base-lr-full", default=DEFAULTS["base_lr_full"], type=float)
    p.add_argument("--base-lr-linear", default=DEFAULTS["base_lr_linear"], type=float)
    p.add_argument("--weight-decay", default=DEFAULTS["weight_decay"], type=float)
    p.add_argument("--warmup-epochs", default=DEFAULTS["warmup_epochs"], type=float)
    p.add_argument("--grad-clip-norm", default=DEFAULTS["grad_clip_norm"], type=float)

    p.add_argument("--early-stop-patience", default=DEFAULTS["early_stop_patience"], type=int)
    p.add_argument("--seed", default=DEFAULTS["random_seed"], type=int)

    return p.parse_args()

# ==================== MAIN ====================
def main():
    args = parse_args()

    TRAIN_DIR = args.train_dir
    VAL_DIR   = args.val_dir
    OUT_DIR   = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    MODEL_NAME  = args.model_name
    NUM_CLASSES = args.num_classes

    EPOCHS        = args.epochs
    FREEZE_EPOCHS = args.freeze_epochs

    BATCH_SIZE  = args.batch_size
    ACCUM_STEPS = args.accum_steps

    LABEL_SMOOTH      = args.label_smooth
    MIXUP_ALPHA       = args.mixup_alpha
    CUTMIX_ALPHA      = args.cutmix_alpha
    MIXUP_PROB        = args.mixup_prob
    MIXUP_SWITCH_PROB = args.mixup_switch_prob

    DROP_RATE      = args.drop_rate
    DROP_PATH_RATE = args.drop_path_rate

    BASE_LR_FULL   = args.base_lr_full
    BASE_LR_LINEAR = args.base_lr_linear
    WEIGHT_DECAY   = args.weight_decay
    WARMUP_EPOCHS  = args.warmup_epochs
    GRAD_CLIP_NORM = args.grad_clip_norm

    EARLY_STOP_PATIENCE = args.early_stop_patience
    RANDOM_SEED         = args.seed

    print("CUDA:", torch.version.cuda if torch.cuda.is_available() else "none",
          "| PyTorch:", torch.__version__)
    set_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(num_classes=NUM_CLASSES, model_name=MODEL_NAME,
                        drop_rate=DROP_RATE, drop_path_rate=DROP_PATH_RATE).to(device)
    cfg = resolve_model_data_config(model)
    img_size = cfg["input_size"][-1]; mean, std = cfg["mean"], cfg["std"]

    train_ds = AlbumentationsImageFolder(root=TRAIN_DIR, transform=make_train_tfms(img_size, mean, std))
    val_ds   = AlbumentationsImageFolder(root=VAL_DIR,   transform=make_val_tfms(img_size, mean, std))

    assert len(train_ds.classes) == NUM_CLASSES, f"Train classes={len(train_ds.classes)} != NUM_CLASSES={NUM_CLASSES}"
    assert train_ds.classes == val_ds.classes, "Class order/names differ between train and val"
    print(f"✔ Classes: {NUM_CLASSES}")
    print(f"✔ Train imgs: {len(train_ds)} | Val imgs: {len(val_ds)}")
    train_counts = [0]*NUM_CLASSES; val_counts = [0]*NUM_CLASSES
    for _, y in train_ds.samples: train_counts[y] += 1
    for _, y in val_ds.samples:   val_counts[y]   += 1
    for i, cls in enumerate(train_ds.classes):
        print(f"  {i:02d}: {cls} | Train {train_counts[i]:5d} | Val {val_counts[i]:5d}")
    assert sum(train_counts) == len(train_ds) and sum(val_counts) == len(val_ds), "Per-class counts mismatch sums"

    train_loader = make_loader(ds=train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(ds=val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model_ema = ModelEmaV2(model=model, decay=0.9999, device=device)
    mixup_fn = Mixup(
        mixup_alpha=MIXUP_ALPHA,
        cutmix_alpha=CUTMIX_ALPHA,
        prob=MIXUP_PROB,
        switch_prob=MIXUP_SWITCH_PROB,
        mode="batch",
        correct_lam=True,
        label_smoothing=LABEL_SMOOTH,
        num_classes=NUM_CLASSES
    )
    criterion = SoftTargetCrossEntropy()

    set_backbone_trainable(m=model, trainable=False if FREEZE_EPOCHS > 0 else True)
    total_params, trainable_params = count_params(model)
    print(f"✔ Model {MODEL_NAME} | Params total={total_params:,} | trainable={trainable_params:,}")

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=BASE_LR_LINEAR if FREEZE_EPOCHS > 0 else BASE_LR_FULL,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    scaler = GradScaler(enabled=torch.cuda.is_available())

    steps_per_epoch = len(train_loader)
    total_steps     = EPOCHS * max(1, steps_per_epoch)
    warmup_steps    = max(1, int(WARMUP_EPOCHS * max(1, steps_per_epoch)))
    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=total_steps - warmup_steps,
        lr_min=1e-6,
        warmup_t=warmup_steps,
        warmup_lr_init=1e-6
    )

    best_acc = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    steps = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        for i, (images, targets_hard) in enumerate(train_loader):
            assert isinstance(images, torch.Tensor) and isinstance(targets_hard, torch.Tensor)
            assert images.ndim == 4 and images.size(0) <= BATCH_SIZE, "Unexpected batch shape/size"
            assert targets_hard.ndim == 1, "Targets must be 1D indices"

            images = images.to(device, non_blocking=True)
            targets_hard = targets_hard.to(device, non_blocking=True)

            hard_for_acc = targets_hard.clone()
            images, targets_soft = mixup_fn(images, targets_hard)

            with disable_all_checkpointing():
                with autocast_ctx():
                    logits = model(images)
                    assert logits.shape[1] == NUM_CLASSES, f"Logits dim {logits.shape[1]} != NUM_CLASSES {NUM_CLASSES}"
                    loss = criterion(logits, targets_soft)

            scaler.scale(loss).backward()
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step_update(num_updates=steps)
                steps += 1
                model_ema.update(model)

            running_loss += loss.item() * images.size(0)
            preds = logits.softmax(1).argmax(1)
            running_correct += (preds == hard_for_acc).sum().item()
            running_total += images.size(0)

        if (len(train_loader) % ACCUM_STEPS) != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step_update(num_updates=steps)
            steps += 1
            model_ema.update(model)

        train_loss = running_loss / max(1, running_total)
        train_acc  = 100.0 * running_correct / max(1, running_total)

        if FREEZE_EPOCHS > 0 and epoch == FREEZE_EPOCHS:
            set_backbone_trainable(m=model, trainable=True)
            for g in optimizer.param_groups:
                g['lr'] = BASE_LR_FULL
            total_params, trainable_params = count_params(model)
            print(f"→ Unfroze backbone; LR -> {BASE_LR_FULL:g} | trainable params={trainable_params:,}")

        metrics = evaluate(model=model_ema.module, loader=val_loader, device=device, num_classes=NUM_CLASSES)
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.2f}% | "
            f"val loss {metrics['loss']:.4f} acc {metrics['acc']:.2f}% | "
            f"Prec {metrics['prec']:.2f}% | Rec {metrics['rec']:.2f}% | "
            f"F1 {metrics['f1']:.2f}% | Top1 {metrics['top1']:.2f}% | Top5 {metrics['top5']:.2f}%"
        )

        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            save_path = os.path.join(OUT_DIR, "best_Deit_XL_Stratified.pth")
            torch.save({
                "model": model_ema.module.state_dict(),
                "acc": best_acc,
                "epoch": epoch,
                "classes": NUM_CLASSES,
                "class_names": train_ds.classes,
                "model_name": MODEL_NAME,
                "img_size": img_size,
                "cfg": cfg,
                "conf_matrix": metrics["cm"]
            }, save_path)
            print(f"  ↳ Saved new BEST acc: {best_acc:.2f}% → {save_path}")

        if metrics["loss"] + 1e-4 < best_val_loss:
            best_val_loss = metrics["loss"]; epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping (no val loss improvement for {EARLY_STOP_PATIENCE} epochs).")
                break

    print(f"Done. Best val acc: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
