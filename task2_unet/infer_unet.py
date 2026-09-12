"""
COMP3710 Lab 2 - Task 4.4.2: Live inference demo script.

Loads a trained UNet checkpoint and runs inference on a single test-set
image, printing per-class DSC for that slice and saving a comparison
image. Designed to run in a few seconds so it can be executed live
during the demonstration (works on CPU or GPU).

Usage:
    python infer_unet.py --run_name unet_baseline --checkpoint unet_best.pt
    python infer_unet.py --run_name unet_baseline --checkpoint unet_best.pt --slice case_412_slice_15.nii.png
"""

import os
import glob
import argparse
import random

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_unet import UNet, NUM_CLASSES, CLASS_NAMES, MASK_DIVISOR, hard_dsc

DATA_ROOT = "/home/groups/comp3710/OASIS"
TEST_IMG_DIR = os.path.join(DATA_ROOT, "keras_png_slices_test")
TEST_SEG_DIR = os.path.join(DATA_ROOT, "keras_png_slices_seg_test")

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default="unet_best.pt")
parser.add_argument("--slice", type=str, default=None,
                     help="Specific case_XXX_slice_N.nii.png filename; "
                          "random test slice if omitted")
args = parser.parse_args()

CHECKPOINT_PATH = os.path.expanduser(
    f"~/comp3710_lab2/outputs/{args.run_name}/{args.checkpoint}")
OUTPUT_PATH = os.path.expanduser(
    f"~/comp3710_lab2/outputs/{args.run_name}/live_inference.png")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Device: {device}")
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    model = UNet().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    if args.slice:
        img_path = os.path.join(TEST_IMG_DIR, args.slice)
    else:
        candidates = glob.glob(os.path.join(TEST_IMG_DIR, "*.png"))
        img_path = random.choice(candidates)

    seg_name = os.path.basename(img_path).replace("case_", "seg_")
    seg_path = os.path.join(TEST_SEG_DIR, seg_name)

    print(f"Running inference on: {os.path.basename(img_path)}")

    to_tensor = transforms.ToTensor()
    img = Image.open(img_path).convert("L")
    img_t = to_tensor(img).unsqueeze(0).to(device)  # [1, 1, 256, 256]

    mask = Image.open(seg_path).convert("L")
    mask_arr = np.array(mask) // MASK_DIVISOR
    mask_t = torch.from_numpy(mask_arr).long().unsqueeze(0).to(device)  # [1, 256, 256]

    with torch.no_grad():
        logits = model(img_t)
        pred = torch.argmax(logits, dim=1)
        dsc = hard_dsc(logits, mask_t, NUM_CLASSES).cpu().numpy()

    print("\nPer-class DSC for this slice:")
    for name, val in zip(CLASS_NAMES, dsc):
        status = "PASS" if val > 0.9 else "FAIL"
        print(f"  {name:15s}: {val:.4f}  [{status}, threshold 0.9]")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    axes[0].imshow(img_t[0, 0].cpu().numpy(), cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(mask_t[0].cpu().numpy(), cmap="viridis", vmin=0, vmax=3)
    axes[1].set_title("Ground truth")
    axes[1].axis("off")
    axes[2].imshow(pred[0].cpu().numpy(), cmap="viridis", vmin=0, vmax=3)
    axes[2].set_title("Prediction")
    axes[2].axis("off")
    plt.suptitle(os.path.basename(img_path))
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nSaved comparison image to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()