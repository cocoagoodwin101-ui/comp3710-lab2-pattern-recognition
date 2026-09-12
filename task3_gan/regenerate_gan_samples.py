"""
COMP3710 Lab 2 - Task 3: regenerate GAN sample images from an existing
checkpoint after fixing the visualization bug in train_gan_oasis.py.

No retraining needed - the generator/critic weights were never wrong,
only the grid-tiling code that rendered them.
"""

import os
import argparse

import torch

from train_gan_oasis import Generator, LATENT_DIM, IMG_SIZE, _save_grid

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")
args = parser.parse_args()

OUTPUT_DIR = os.path.expanduser(f"~/comp3710_lab2/outputs/{args.run_name}")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, args.checkpoint)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    generator = Generator().to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    fixed_noise = ckpt["fixed_noise"].to(device)
    epoch = ckpt["epoch"]
    print(f"Checkpoint is from epoch {epoch}")

    with torch.no_grad():
        fixed_fakes = ((generator(fixed_noise) + 1) / 2).cpu()
    _save_grid(fixed_fakes, 8,
               os.path.join(OUTPUT_DIR, f"CORRECTED_fixed_epoch_{epoch}.png"),
               f"Fixed-noise samples, epoch {epoch} (corrected)")

    random_noise = torch.randn(64, LATENT_DIM, device=device)
    with torch.no_grad():
        random_fakes = ((generator(random_noise) + 1) / 2).cpu()
    _save_grid(random_fakes, 8,
               os.path.join(OUTPUT_DIR, "CORRECTED_random_samples.png"),
               "Random-noise samples (corrected, diversity/mode-collapse check)")

    print("Saved CORRECTED_fixed_epoch_*.png and CORRECTED_random_samples.png to",
          OUTPUT_DIR)


if __name__ == "__main__":
    main()