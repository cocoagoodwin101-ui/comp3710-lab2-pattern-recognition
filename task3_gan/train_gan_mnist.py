"""
COMP3710 Lab 2 - Task 4.4 Task 3 (Hard tier), Phase A: WGAN-GP on MNIST.

This is a throwaway smoke test, NOT the graded deliverable. Purpose:
validate the WGAN-GP training loop (critic/generator update ratio,
gradient penalty, loss behaviour) on a small, fast dataset before
touching OASIS, where a bug would be much more expensive to discover
(slower iteration, less obvious "it's clearly not working" signal
than MNIST digits either forming or not).

WGAN-GP summary:
  - Critic (not "discriminator" - outputs a raw score, no sigmoid)
    loss: maximise E[critic(real)] - E[critic(fake)], i.e. minimise
    -(critic_real - critic_fake) + lambda * gradient_penalty.
  - Generator loss: maximise E[critic(fake)], i.e. minimise
    -critic(fake).
  - Gradient penalty enforces the critic stays ~1-Lipschitz, which the
    Wasserstein-distance approximation requires to be valid.
  - Critic trained N_CRITIC steps per generator step (paper default 5)
    so the critic stays close to optimal.
  - No BatchNorm in the critic (interacts badly with the per-sample
    gradient penalty); InstanceNorm used instead as a substitute for
    LayerNorm in the conv critic.
  - Adam with beta1=0, beta2=0.9 (WGAN-GP paper's recommendation -
    notably different from the beta1=0.5 typical of vanilla DCGAN).

Runs on the cpu partition since MNIST at 32x32 doesn't need a GPU -
avoids the comp3710 GPU queue entirely for this prototype step.
"""

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------
# CLI args
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default="gan_mnist")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--n_critic", type=int, default=5)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lambda_gp", type=float, default=10.0)
parser.add_argument("--sample_every", type=int, default=2,
                     help="Save a fixed-noise sample grid every N epochs")
parser.add_argument("--subset_size", type=int, default=None,
                     help="Use only this many MNIST images (for a fast smoke test). "
                          "Omit to use the full 60000.")
args = parser.parse_args()

OUTPUT_DIR = os.path.expanduser(f"~/comp3710_lab2/outputs/{args.run_name}")
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

IMG_SIZE = 32  # resized up from MNIST's native 28x28 for clean conv arithmetic
LATENT_DIM = args.latent_dim
N_CRITIC = args.n_critic
LAMBDA_GP = args.lambda_gp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    torch.set_num_threads(n_cpus)
    print(f"CPU mode: using {n_cpus} threads")


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # -> [-1, 1] to match Tanh generator output
])


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, img_channels=1, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base * 4, 4, 1, 0, bias=False),  # 1->4
            nn.BatchNorm2d(base * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False),    # 4->8
            nn.BatchNorm2d(base * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False),       # 8->16
            nn.BatchNorm2d(base),
            nn.ReLU(True),
            nn.ConvTranspose2d(base, img_channels, 4, 2, 1, bias=False),   # 16->32
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, img_channels=1, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, base, 4, 2, 1),          # 32->16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1),               # 16->8
            nn.InstanceNorm2d(base * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),           # 8->4
            nn.InstanceNorm2d(base * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, 1, 4, 1, 0),                  # 4->1
        )

    def forward(self, x):
        return self.net(x).view(-1)


# ----------------------------------------------------------------------
# Gradient penalty
# ----------------------------------------------------------------------
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
          f"n_critic={N_CRITIC} | lr={args.lr} | lambda_gp={LAMBDA_GP}")
    print(f"Device: {device}")

    dataset = torchvision.datasets.MNIST(
        root=os.path.expanduser("~/comp3710_lab2/data"),
        train=True, download=True, transform=transform,
    )
    if args.subset_size is not None:
        indices = torch.randperm(len(dataset))[:args.subset_size]
        dataset = torch.utils.data.Subset(dataset, indices.tolist())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=2, drop_last=True)
    print(f"Training images: {len(dataset)}")

    generator = Generator().to(device)
    critic = Critic().to(device)

    gen_opt = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.9))
    critic_opt = optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.9))

    fixed_noise = torch.randn(64, LATENT_DIM, device=device)  # same noise every
                                                                # sample grid, so we
                                                                # can see the SAME
                                                                # generated digits
                                                                # improve over epochs

    history = {"critic_loss": [], "gen_loss": [], "wasserstein_est": []}

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        epoch_critic_loss = epoch_gen_loss = epoch_wdist = 0.0
        n_batches = 0

        for real, _ in loader:
            real = real.to(device)
            batch_size = real.size(0)

            # --- critic: N_CRITIC updates per generator update ---
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

            # --- generator: one update ---
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

    torch.save(generator.state_dict(), os.path.join(OUTPUT_DIR, "generator.pt"))
    torch.save(critic.state_dict(), os.path.join(OUTPUT_DIR, "critic.pt"))
    np.savez(os.path.join(OUTPUT_DIR, "gan_history.npz"), **history)

    plot_loss_curves(history)
    print("Done. Outputs saved to", OUTPUT_DIR)


def save_sample_grid(generator, fixed_noise, epoch, nrow=8):
    generator.eval()
    with torch.no_grad():
        fakes = generator(fixed_noise).cpu()
    fakes = (fakes + 1) / 2  # [-1,1] -> [0,1] for display

    grid = fakes.reshape(nrow, nrow, IMG_SIZE, IMG_SIZE)
    canvas = np.zeros((nrow * IMG_SIZE, nrow * IMG_SIZE))
    for i in range(nrow):
        for j in range(nrow):
            canvas[i * IMG_SIZE:(i + 1) * IMG_SIZE,
                   j * IMG_SIZE:(j + 1) * IMG_SIZE] = grid[i, j, 0]

    plt.figure(figsize=(6, 6))
    plt.imshow(canvas, cmap="gray")
    plt.title(f"Fixed-noise samples, epoch {epoch}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(SAMPLES_DIR, f"epoch_{epoch:04d}.png"), dpi=150)
    plt.close()
    generator.train()


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