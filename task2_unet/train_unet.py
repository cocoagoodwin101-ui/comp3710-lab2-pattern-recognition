"""
COMP3710 Lab 2 - Task 4.4.2: UNet segmentation on OASIS brain MRI slices.

Trains a UNet on 256x256 grayscale brain MRI slices to segment 4 classes
(background, CSF, grey matter, white matter -- standard OASIS 4-class
convention; values in the raw masks are {0, 85, 170, 255}, mapped here
to class indices {0, 1, 2, 3}).

Loss: combined soft-Dice + cross-entropy. Dice is used because it's the
same overlap quantity as the DSC evaluation metric, so training loss and
the grading metric are directly aligned; cross-entropy is added because
Dice gradients are noisy early in training when overlap is near zero.

Evaluation: per-class DSC computed on hard (argmax) predictions each
epoch on the validation set -- this is the number that has to clear 0.9
for every class, so we track it directly rather than only the loss.

Run on Rangpur via the accompanying job_unet.sh sbatch script.
"""

import os
import glob
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------
# CLI args
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default="unet_baseline")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--base_channels", type=int, default=32)
args, _ = parser.parse_known_args()


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
DATA_ROOT = "/home/groups/comp3710/OASIS"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "keras_png_slices_train")
TRAIN_SEG_DIR = os.path.join(DATA_ROOT, "keras_png_slices_seg_train")
VALID_IMG_DIR = os.path.join(DATA_ROOT, "keras_png_slices_validate")
VALID_SEG_DIR = os.path.join(DATA_ROOT, "keras_png_slices_seg_validate")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "keras_png_slices_test")
TEST_SEG_DIR = os.path.join(DATA_ROOT, "keras_png_slices_seg_test")

OUTPUT_DIR = os.path.expanduser(f"~/comp3710_lab2/outputs/{args.run_name}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 256
NUM_CLASSES = 4
CLASS_NAMES = ["background", "CSF", "grey_matter", "white_matter"]  #csf = cerebrospinal fluid
# standard OASIS 4-class convention; verify against dataset docs if precise anatomical labelling is asked
MASK_DIVISOR = 85  # raw mask values {0,85,170,255} -> class indices {0,1,2,3}

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.lr
BASE_CHANNELS = args.base_channels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class OASISSegDataset(Dataset):
    """Loads paired (image, mask) tensors. Filenames match via
    case_XXX_slice_N.nii.png <-> seg_XXX_slice_N.nii.png.

    All images/masks are decoded from disk ONCE here in __init__ and
    kept in memory as tensors -- avoids re-reading and re-decoding
    thousands of PNGs from the shared filesystem on every single epoch.
    ~9664 train images at 256x256 uint8 is ~630MB, comfortably fits in
    RAM; kept on CPU here, existing per-batch .to(device) calls handle
    GPU staging as before."""

    def __init__(self, img_dir, seg_dir):
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if len(img_files) == 0:
            raise RuntimeError(f"No PNG files found in {img_dir}")

        to_tensor = transforms.ToTensor()
        print(f"Preloading {len(img_files)} image/mask pairs from {img_dir} ...")
        t0 = time.time()

        self.images = []
        self.masks = []
        for img_path in img_files:
            seg_name = os.path.basename(img_path).replace("case_", "seg_")
            seg_path = os.path.join(seg_dir, seg_name)

            img = Image.open(img_path).convert("L")
            self.images.append(to_tensor(img))  # [1, 256, 256] in [0, 1]

            mask = Image.open(seg_path).convert("L")
            mask_arr = np.array(mask) // MASK_DIVISOR  # -> {0,1,2,3}
            self.masks.append(torch.from_numpy(mask_arr).long())  # [256, 256]

        print(f"Preload done in {time.time() - t0:.1f}s")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

# the repeated building blocks 3x3 with stride = 1, keeps spatial size unchanged never changes resolution.
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=NUM_CLASSES, base=BASE_CHANNELS):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out_conv = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)               # 256x256
        e2 = self.enc2(self.pool(e1))   # 128x128
        e3 = self.enc3(self.pool(e2))   # 64x64
        e4 = self.enc4(self.pool(e3))   # 32x32
        b = self.bottleneck(self.pool(e4))  # 16x16

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)  # logits [B, num_classes, 256, 256]


# ----------------------------------------------------------------------
# Loss and metrics
# ----------------------------------------------------------------------
def soft_dice_loss(logits, targets, num_classes, eps=1e-6):
    """Differentiable Dice computed on softmax probabilities (for training)."""
    probs = F.softmax(logits, dim=1)
    targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets_onehot, dims)
    cardinality = torch.sum(probs + targets_onehot, dims)
    dice_per_class = (2 * intersection + eps) / (cardinality + eps)
    return 1 - dice_per_class.mean()


def combined_loss(logits, targets, num_classes):
    ce = F.cross_entropy(logits, targets)
    dice = soft_dice_loss(logits, targets, num_classes)
    return ce + dice


def hard_dsc(logits, targets, num_classes, eps=1e-6):
    """DSC per class from argmax (hard) predictions -- the actual metric
    the lab sheet's >0.9 requirement applies to."""
    preds = torch.argmax(logits, dim=1)
    preds_onehot = F.one_hot(preds, num_classes).permute(0, 3, 1, 2).float()
    targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(preds_onehot * targets_onehot, dims)
    cardinality = torch.sum(preds_onehot + targets_onehot, dims)
    return (2 * intersection + eps) / (cardinality + eps)  # [num_classes]

# DSC Formula: DSC = 2·|X ∩ Y| / (|X| + |Y|), where X = predicted pixels for a class, 
# Y = true pixels for that class. Ranges 0 (no overlap) to 1 (perfect match).
# eps prevents division by zero error if some class never appears at all in a given batch


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train():
    print(f"Run: {args.run_name} | epochs={EPOCHS} | batch_size={BATCH_SIZE} | "
          f"lr={LR} | base_channels={BASE_CHANNELS}")
    print(f"Device: {device}")

    train_ds = OASISSegDataset(TRAIN_IMG_DIR, TRAIN_SEG_DIR)
    valid_ds = OASISSegDataset(VALID_IMG_DIR, VALID_SEG_DIR)
    print(f"Train slices: {len(train_ds)}, Validation slices: {len(valid_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=4, pin_memory=True)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "val_dsc": []}  # val_dsc: list of
                                                                   # per-class arrays

    best_mean_dsc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start = time.time()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = combined_loss(logits, masks, NUM_CLASSES)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_ds)

        model.eval()
        val_running = 0.0
        dsc_sum = torch.zeros(NUM_CLASSES).to(device)
        n_batches = 0
        with torch.no_grad():
            for imgs, masks in valid_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = combined_loss(logits, masks, NUM_CLASSES)
                val_running += loss.item() * imgs.size(0)
                dsc_sum += hard_dsc(logits, masks, NUM_CLASSES)
                n_batches += 1
        val_loss = val_running / len(valid_ds)
        val_dsc = (dsc_sum / n_batches).cpu().numpy()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dsc"].append(val_dsc)

        mean_dsc = val_dsc.mean()
        if mean_dsc > best_mean_dsc:
            best_mean_dsc = mean_dsc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "unet_best.pt"))

        elapsed = time.time() - start
        dsc_str = ", ".join(f"{n}={v:.3f}" for n, v in zip(CLASS_NAMES, val_dsc))
        print(f"Epoch {epoch:3d}/{EPOCHS} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | DSC: {dsc_str} | {elapsed:.1f}s")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "unet_final.pt"))
    np.savez(os.path.join(OUTPUT_DIR, "unet_history.npz"),
              train_loss=history["train_loss"],
              val_loss=history["val_loss"],
              val_dsc=np.array(history["val_dsc"]))

    plot_loss_curve(history)
    plot_dsc_curve(history)
    plot_predictions(model, valid_loader)

    final_dsc = history["val_dsc"][-1]
    print("\nFinal per-class DSC (last epoch):")
    for name, val in zip(CLASS_NAMES, final_dsc):
        status = "PASS" if val > 0.9 else "FAIL"
        print(f"  {name:15s}: {val:.4f}  [{status}, threshold 0.9]")
    print(f"\nBest mean DSC during training: {best_mean_dsc:.4f} "
          f"(checkpoint saved as unet_best.pt)")
    print("Done. Outputs saved to", OUTPUT_DIR)


# ----------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------
def plot_loss_curve(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Dice + CE)")
    plt.title("UNet Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=150)
    plt.close()


def plot_dsc_curve(history):
    dsc_arr = np.array(history["val_dsc"])  # [epochs, num_classes]
    plt.figure(figsize=(8, 5))
    for i, name in enumerate(CLASS_NAMES):
        plt.plot(dsc_arr[:, i], label=name)
    plt.axhline(0.9, color="red", linestyle="--", label="0.9 threshold")
    plt.xlabel("Epoch")
    plt.ylabel("DSC")
    plt.title("Per-class Validation DSC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "dsc_curve.png"), dpi=150)
    plt.close()


def plot_predictions(model, loader, n=6):
    """Original image / ground truth mask / predicted mask, side by side,
    for a handful of validation slices -- justifies the DSC numbers
    visually rather than just quoting them."""
    model.eval()
    imgs, masks = next(iter(loader))
    imgs, masks = imgs[:n].to(device), masks[:n].to(device)
    with torch.no_grad():
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)

    fig, axes = plt.subplots(3, n, figsize=(2.2 * n, 6.5))
    for i in range(n):
        axes[0, i].imshow(imgs[i, 0].cpu().numpy(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(masks[i].cpu().numpy(), cmap="viridis", vmin=0, vmax=3)
        axes[1, i].axis("off")
        axes[2, i].imshow(preds[i].cpu().numpy(), cmap="viridis", vmin=0, vmax=3)
        axes[2, i].axis("off")
    axes[0, 0].set_title("Image", loc="left")
    axes[1, 0].set_title("Ground truth", loc="left")
    axes[2, 0].set_title("Prediction", loc="left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "predictions.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    train()