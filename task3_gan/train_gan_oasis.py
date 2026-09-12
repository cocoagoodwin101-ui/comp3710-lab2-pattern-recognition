"""
COMP3710 Lab 2 - Task 4.4 Task 3 (Hard tier), Phase B: WGAN-GP on OASIS.

Ported from the MNIST prototype (train_gan_mnist.py) after confirming
the training loop is numerically stable there (bounded losses, no
NaN/explosion). Scaled to 64x64 grayscale brain slices, unconditional
(no segmentation masks needed).

Supports checkpoint/resume so training can span multiple sbatch
submissions if one job's time limit isn't enough for convincing
results -- pass --resume to continue from the last saved checkpoint
in this run's output folder.
"""

import os
import glob
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
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
parser.add_argument("--run_name", type=str, default="gan_oasis")
parser.add_argument("--epochs", type=int, default=100,
                     help="Total epochs to reach (not additional epochs when resuming)")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--n_critic", type=int, default=5)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lambda_gp", type=float, default=10.0)
parser.add_argument("--sample_every", type=int, default=5)
parser.add_argument("--checkpoint_every", type=int, default=10)
parser.add_argument("--resume", action="store_true",
                     help="Resume from checkpoint.pt in this run's output folder")
args = parser.parse_args()

DATA_ROOT = "/home/groups/comp3710/OASIS"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "keras_png_slices_train")

OUTPUT_DIR = os.path.expanduser(f"~/comp3710_lab2/outputs/{args.run_name}")
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pt")

IMG_SIZE = 64
LATENT_DIM = args.latent_dim
N_CRITIC = args.n_critic
LAMBDA_GP = args.lambda_gp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Dataset - preloaded into memory once, same lesson as the UNet fix
# ----------------------------------------------------------------------
class OASISGanDataset(Dataset):
    def __init__(self, img_dir):
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if len(img_files) == 0:
            raise RuntimeError(f"No PNG files found in {img_dir}")

        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # -> [-1, 1] for Tanh generator output
        ])

        print(f"Preloading {len(img_files)} images from {img_dir} ...")
        t0 = time.time()
        self.images = []
        for path in img_files:
            img = Image.open(path).convert("L")
            self.images.append(transform(img))
        print(f"Preload done in {time.time() - t0:.1f}s")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


# ----------------------------------------------------------------------
# Models - 64x64, one more up/downsampling stage than the MNIST version
# ----------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, img_channels=1, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base * 8, 4, 1, 0, bias=False),  # 1->4
            nn.BatchNorm2d(base * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1, bias=False),    # 4->8
            nn.BatchNorm2d(base * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False),    # 8->16
            nn.BatchNorm2d(base * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False),        # 16->32
            nn.BatchNorm2d(base),
            nn.ReLU(True),
            nn.ConvTranspose2d(base, img_channels, 4, 2, 1, bias=False),    # 32->64
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, img_channels=1, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, base, 4, 2, 1),           # 64->32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1),                # 32->16
            nn.InstanceNorm2d(base * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),            # 16->8
            nn.InstanceNorm2d(base * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 8, 4, 2, 1),            # 8->4
            nn.InstanceNorm2d(base * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 8, 1, 4, 1, 0),                   # 4->1
        )

    def forward(self, x):
        return self.net(x).view(-1)


def gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolates,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train():
    print(f"Run: {args.run_name} | epochs={args.epochs} | batch_size={args.batch_size} | "
          f"n_critic={N_CRITIC} | lr={args.lr} | lambda_gp={LAMBDA_GP} | resume={args.resume}")
    print(f"Device: {device}")

    dataset = OASISGanDataset(TRAIN_IMG_DIR)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=4, pin_memory=True, drop_last=True)
    print(f"Training images: {len(dataset)}")

    generator = Generator().to(device)
    critic = Critic().to(device)
    gen_opt = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.9))
    critic_opt = optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.9))

    fixed_noise = torch.randn(64, LATENT_DIM, device=device)

    start_epoch = 1
    history = {"critic_loss": [], "gen_loss": [], "wasserstein_est": []}

    if args.resume and os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        generator.load_state_dict(ckpt["generator"])
        critic.load_state_dict(ckpt["critic"])
        gen_opt.load_state_dict(ckpt["gen_opt"])
        critic_opt.load_state_dict(ckpt["critic_opt"])
        fixed_noise = ckpt["fixed_noise"].to(device)
        start_epoch = ckpt["epoch"] + 1
        history = ckpt["history"]
        print(f"Resumed at epoch {start_epoch}")
    elif args.resume:
        print("--resume requested but no checkpoint found, starting fresh")

    if start_epoch > args.epochs:
        print(f"Already at epoch {start_epoch - 1}, target {args.epochs} reached. Nothing to do.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        epoch_critic_loss = epoch_gen_loss = epoch_wdist = 0.0
        n_batches = 0

        for real in loader:
            real = real.to(device)
            batch_size = real.size(0)

            for _ in range(N_CRITIC):
                noise = torch.randn(batch_size, LATENT_DIM, device=device)
                fake = generator(noise).detach()

                critic_real = critic(real).mean()
                critic_fake = critic(fake).mean()
                gp = gradient_penalty(critic, real, fake, device)
                critic_loss = -(critic_real - critic_fake) + LAMBDA_GP * gp

                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            fake = generator(noise)
            gen_loss = -critic(fake).mean()

            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

            epoch_critic_loss += critic_loss.item()
            epoch_gen_loss += gen_loss.item()
            epoch_wdist += (critic_real - critic_fake).item()
            n_batches += 1

        history["critic_loss"].append(epoch_critic_loss / n_batches)
        history["gen_loss"].append(epoch_gen_loss / n_batches)
        history["wasserstein_est"].append(epoch_wdist / n_batches)

        elapsed = time.time() - start
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"critic_loss={history['critic_loss'][-1]:.4f} | "
              f"gen_loss={history['gen_loss'][-1]:.4f} | "
              f"W_est={history['wasserstein_est'][-1]:.4f} | {elapsed:.1f}s")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            save_sample_grid(generator, fixed_noise, epoch)

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            torch.save({
                "generator": generator.state_dict(),
                "critic": critic.state_dict(),
                "gen_opt": gen_opt.state_dict(),
                "critic_opt": critic_opt.state_dict(),
                "fixed_noise": fixed_noise.cpu(),
                "epoch": epoch,
                "history": history,
            }, CHECKPOINT_PATH)
            print(f"Checkpoint saved at epoch {epoch}")

    torch.save(generator.state_dict(), os.path.join(OUTPUT_DIR, "generator_final.pt"))
    torch.save(critic.state_dict(), os.path.join(OUTPUT_DIR, "critic_final.pt"))
    np.savez(os.path.join(OUTPUT_DIR, "gan_history.npz"), **history)

    plot_loss_curves(history)
    save_random_sample_grid(generator)  # diversity check, different noise each cell
    print("Done. Outputs saved to", OUTPUT_DIR)


def save_sample_grid(generator, fixed_noise, epoch, nrow=8):
    """Same fixed noise every time -- shows the SAME 64 slots sharpening
    over training, i.e. progression."""
    generator.eval()
    with torch.no_grad():
        fakes = generator(fixed_noise).cpu()
    fakes = (fakes + 1) / 2
    _save_grid(fakes, nrow, os.path.join(SAMPLES_DIR, f"epoch_{epoch:04d}.png"),
               f"Fixed-noise samples, epoch {epoch}")
    generator.train()


def save_random_sample_grid(generator, nrow=8):
    """Fresh random noise -- this is the diversity/mode-collapse check:
    64 different-looking brains means no collapse, 64 near-identical
    ones means collapse."""
    generator.eval()
    noise = torch.randn(nrow * nrow, LATENT_DIM, device=device)
    with torch.no_grad():
        fakes = generator(noise).cpu()
    fakes = (fakes + 1) / 2
    _save_grid(fakes, nrow, os.path.join(OUTPUT_DIR, "random_samples_final.png"),
               "Random-noise samples (diversity / mode-collapse check)")
    generator.train()


def _save_grid(fakes, nrow, path, title):
    grid = fakes.reshape(nrow, nrow, IMG_SIZE, IMG_SIZE)
    canvas = np.zeros((nrow * IMG_SIZE, nrow * IMG_SIZE))
    for i in range(nrow):
        for j in range(nrow):
            canvas[i * IMG_SIZE:(i + 1) * IMG_SIZE,
                   j * IMG_SIZE:(j + 1) * IMG_SIZE] = grid[i, j]
    plt.figure(figsize=(6, 6))
    plt.imshow(canvas, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_loss_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history["critic_loss"], label="Critic loss")
    axes[0].plot(history["gen_loss"], label="Generator loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("WGAN-GP Losses")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["wasserstein_est"], color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("E[critic(real)] - E[critic(fake)]")
    axes[1].set_title("Wasserstein Distance Estimate")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curves.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    train()