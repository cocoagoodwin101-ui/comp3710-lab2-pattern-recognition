"""
COMP3710 Lab 2 - Task 4.4.2: full TEST SET evaluation (not just one slice).

infer_unet.py demonstrates live inference on a single test image, which
satisfies "demonstrate inference on an MRI from the dataset" literally.
This script covers the other reading of the requirement - aggregate
performance across the WHOLE held-out test set - so both are on hand.
"""

import os
import argparse

import torch
from torch.utils.data import DataLoader

from train_unet import (
    UNet, OASISSegDataset, TEST_IMG_DIR, TEST_SEG_DIR,
    NUM_CLASSES, CLASS_NAMES, hard_dsc,
)

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default="unet_best.pt")
parser.add_argument("--batch_size", type=int, default=16)
args, _ = parser.parse_known_args()

OUTPUT_DIR = os.path.expanduser(f"~/comp3710_lab2/outputs/{args.run_name}")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, args.checkpoint)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Device: {device}")
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    test_ds = OASISSegDataset(TEST_IMG_DIR, TEST_SEG_DIR)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Test slices: {len(test_ds)}")

    model = UNet().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    dsc_sum = torch.zeros(NUM_CLASSES).to(device)
    n_batches = 0
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            dsc_sum += hard_dsc(logits, masks, NUM_CLASSES)
            n_batches += 1
    test_dsc = (dsc_sum / n_batches).cpu().numpy()

    print(f"\nFull TEST SET per-class DSC ({len(test_ds)} slices, all of them):")
    for name, val in zip(CLASS_NAMES, test_dsc):
        status = "PASS" if val > 0.9 else "FAIL"
        print(f"  {name:15s}: {val:.4f}  [{status}, threshold 0.9]")


if __name__ == "__main__":
    main()