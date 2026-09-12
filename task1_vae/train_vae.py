"""
COMP3710 Lab 2 - Task 4.4.1: Variational Autoencoder on OASIS brain MRI slices.

Trains a convolutional VAE on 128x128 grayscale brain MRI slices.

Visualisation depends on --latent_dim:
  - latent_dim == 2: decode a grid across the 2D latent space directly
    (the classic VAE manifold plot).
  - latent_dim > 2: (a) encode real validation images and project the
    latent codes to 2D with UMAP for a scatter plot, and (b) linearly
    interpolate between two real images' latent codes and decode each
    step, to show the decoder produces smooth, sensible in-between
    images (a manifold demonstration that doesn't need an invertible
    2D->ND mapping, which UMAP does not provide).

Run on Rangpur via the accompanying job_vae.sh sbatch script.
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
matplotlib.use("Agg")  # no display on a compute node
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------
# CLI args - lets us run isolated, reproducible experiments without
# editing the script each time, and without overwriting prior results.
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default="baseline",
                     help="Subfolder name under outputs/ for this run's results")
parser.add_argument("--beta", type=float, default=1.0, help="KL weight")
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--latent_dim", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
DATA_ROOT = "/home/groups/comp3710/OASIS"
TRAIN_DIR = os.path.join(DATA_ROOT, "keras_png_slices_train")
VALID_DIR = os.path.join(DATA_ROOT, "keras_png_slices_validate")

OUTPUT_DIR = os.path.expanduser(f"~/comp3710_lab2/outputs/{args.run_name}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 128
LATENT_DIM = args.latent_dim
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.lr
BETA = args.beta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class OASISDataset(Dataset):
    """Loads OASIS PNG slices as single-channel tensors in [0, 1]."""

    def __init__(self, root_dir, transform=None):
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        if len(self.files) == 0:
            raise RuntimeError(f"No PNG files found in {root_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")  # ensure single channel
        if self.transform:
            img = self.transform(img)
        return img


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  # scales uint8 [0,255] -> float [0,1]
])


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),    # 128 -> 64
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),   # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 16 -> 8
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 4, stride=2, padding=1), # 8 -> 4
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),  # 4 -> 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 16 -> 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 32 -> 64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),     # 64 -> 128
            nn.Sigmoid(),  # output in [0, 1] to match normalised pixels
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 256, 4, 4)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        # sampling z ~ N(mu, sigma^2) in a way that keeps gradients flowing
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=BETA):
    # reconstruction term: how well we rebuilt the image
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum") / x.size(0)
    # KL term: how close the encoded distribution is to a standard normal
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train():
    print(f"Run: {args.run_name} | beta={BETA} | epochs={EPOCHS} | "
          f"latent_dim={LATENT_DIM} | batch_size={BATCH_SIZE} | lr={LR}")
    print(f"Device: {device}")

    train_ds = OASISDataset(TRAIN_DIR, transform=transform)
    valid_ds = OASISDataset(VALID_DIR, transform=transform)
    print(f"Train slices: {len(train_ds)}, Validation slices: {len(valid_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=4, pin_memory=True)

    model = VAE(LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "train_recon": [], "train_kl": [], "val_loss": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start = time.time()
        running_loss = running_recon = running_kl = 0.0

        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_recon += recon_loss.item() * x.size(0)
            running_kl += kl_loss.item() * x.size(0)

        train_loss = running_loss / len(train_ds)
        train_recon = running_recon / len(train_ds)
        train_kl = running_kl / len(train_ds)

        # quick validation pass
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for x in valid_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                loss, _, _ = vae_loss(recon, x, mu, logvar)
                val_running += loss.item() * x.size(0)
        val_loss = val_running / len(valid_ds)

        history["train_loss"].append(train_loss)
        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - start
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train_loss={train_loss:.2f} (recon={train_recon:.2f}, kl={train_kl:.2f}) | "
              f"val_loss={val_loss:.2f} | {elapsed:.1f}s")

    # save model + loss history
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "vae_model.pt"))
    np.savez(os.path.join(OUTPUT_DIR, "vae_history.npz"), **history)

    plot_loss_curve(history)
    plot_reconstructions(model, valid_loader)

    if LATENT_DIM == 2:
        plot_manifold(model)
    else:
        plot_umap_scatter(model, valid_loader)
        plot_latent_interpolation(model, valid_loader)

    print("Done. Outputs saved to", OUTPUT_DIR)


# ----------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------
def plot_loss_curve(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (recon + KL, per-sample)")
    plt.title("VAE Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=150)
    plt.close()


def plot_reconstructions(model, loader, n=8):
    """Show a handful of original vs reconstructed slices, sanity check."""
    model.eval()
    x = next(iter(loader))[:n].to(device)
    with torch.no_grad():
        recon, _, _ = model(x)

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(x[i, 0].cpu().numpy(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i, 0].cpu().numpy(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
    plt.suptitle("VAE Reconstructions (top: original, bottom: reconstructed)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "reconstructions.png"), dpi=150)
    plt.close()


def plot_manifold(model, grid_size=15, span=3.0):
    """
    Decode a grid of points across the 2D latent space and tile the
    results into one image -- this IS the manifold visualisation.
    span=3.0 covers +/-3 std devs of a standard normal, where the vast
    majority of the encoded training data should live.
    """
    model.eval()
    grid_x = np.linspace(-span, span, grid_size)
    grid_y = np.linspace(-span, span, grid_size)[::-1]  # flip so it reads top-to-bottom

    canvas = np.zeros((grid_size * IMG_SIZE, grid_size * IMG_SIZE))

    with torch.no_grad():
        for i, y in enumerate(grid_y):
            for j, x in enumerate(grid_x):
                z = torch.tensor([[x, y]], dtype=torch.float32).to(device)
                decoded = model.decoder(z).cpu().numpy()[0, 0]
                canvas[i * IMG_SIZE:(i + 1) * IMG_SIZE,
                       j * IMG_SIZE:(j + 1) * IMG_SIZE] = decoded

    plt.figure(figsize=(10, 10))
    plt.imshow(canvas, cmap="gray")
    plt.title("VAE Latent Space Manifold (2D)")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "manifold.png"), dpi=150)
    plt.close()


def plot_umap_scatter(model, loader, n_samples=1500):
    """
    Encode real validation images and project their latent codes down
    to 2D with UMAP. This shows whether the latent space forms a
    connected, sensible structure (good) or scattered disjoint
    clusters/gaps (bad) -- it's a projection of the real structure,
    not a decodable grid like the 2D case.
    """
    import umap  # imported here so the script still runs w/o umap-learn
                 # installed when latent_dim == 2 and this path isn't used

    model.eval()
    codes = []
    collected = 0
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            mu, _ = model.encoder(x)
            codes.append(mu.cpu().numpy())
            collected += x.size(0)
            if collected >= n_samples:
                break
    codes = np.concatenate(codes, axis=0)[:n_samples]

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(codes)

    plt.figure(figsize=(8, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.6)
    plt.title(f"UMAP projection of {LATENT_DIM}D latent space "
              f"({len(codes)} validation slices)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "umap_scatter.png"), dpi=150)
    plt.close()


def plot_latent_interpolation(model, loader, n_pairs=4, n_steps=10):
    """
    Pick a few pairs of real validation images, linearly interpolate
    between their encoded latent means, and decode every step. Each
    row is one interpolation; smooth, sensible in-between brains
    (rather than noise or an abrupt jump) is the manifold evidence
    for a latent space too large to grid-visualise directly.
    """
    model.eval()
    x = next(iter(loader))[: n_pairs * 2].to(device)
    with torch.no_grad():
        mu, _ = model.encoder(x)

    fig, axes = plt.subplots(n_pairs, n_steps, figsize=(1.5 * n_steps, 1.5 * n_pairs))
    with torch.no_grad():
        for row in range(n_pairs):
            z_start = mu[2 * row]
            z_end = mu[2 * row + 1]
            for col, alpha in enumerate(np.linspace(0, 1, n_steps)):
                z = (1 - alpha) * z_start + alpha * z_end
                decoded = model.decoder(z.unsqueeze(0)).cpu().numpy()[0, 0]
                axes[row, col].imshow(decoded, cmap="gray")
                axes[row, col].axis("off")

    plt.suptitle(f"Latent-space interpolation ({LATENT_DIM}D), "
                  f"{n_pairs} random real-image pairs")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "latent_interpolation.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    train()