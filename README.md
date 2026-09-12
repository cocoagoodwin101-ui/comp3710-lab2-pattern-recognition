# COMP3710 Pattern Analysis - Lab 2: Pattern Recognition

Lab Demonstration 2 submission. This lab covers dimensionality reduction,
classification, and deep learning pipelines using PyTorch on UQ's Rangpur
HPC cluster (comp3710 partition, A100 GPUs).

## Repository Structure

```
Lab2/
├── README.md
├── .gitignore
├── task1_vae/              # Section 4.4, Task 1
│   ├── train_vae.py
│   ├── job_vae.sh
│   └── results/
├── task2_unet/              # Section 4.4, Task 2
│   ├── train_unet.py
│   ├── infer_unet.py
│   ├── job_unet.sh
│   ├── job_unet_infer.sh
│   └── results/
└── task3_gan/                # Section 4.4, Task 3 (Hard tier)
    ├── train_gan_mnist.py    # Phase A: MNIST prototype
    ├── job_gan_mnist.sh
    ├── train_gan_oasis.py    # Phase B: OASIS training
    ├── job_gan_oasis.sh
    ├── regenerate_gan_samples.py
    ├── job_gan_regen.sh
    └── results/
```

> **TODO (not yet added):** Part 1 (DFT/Fourier), Part 2 (Eigenfaces/PCA +
> Random Forest), Part 3.1 (CNN classifier), Part 3.2 (DAWNBench ResNet-18
> challenge), Advanced Git course evidence. Folders and sections below for
> these will be added as they're completed.

## Environment

- **Cluster:** Rangpur HPC, `comp3710` partition (A100 GPUs), jobs submitted
  via `sbatch`.
- **Framework:** PyTorch, conda environment `torch` (Python 3.11).
- **Dataset:** OASIS preprocessed brain MRI slices, `/home/groups/comp3710/OASIS`
  (256x256 grayscale PNGs; segmentation masks with 4 classes: background,
  CSF, grey matter, white matter).

---

## Section 4.4, Task 1: Variational Autoencoder (VAE)

**Goal:** train a VAE on OASIS brain slices and visualise the learned
latent manifold.

**Approach:** three deliberate iterations rather than a single run:
1. **Baseline** — 2D latent space (chosen specifically so the manifold
   could be visualised as a directly-decoded image grid), 128x128
   resolution, standard VAE loss (BCE + KL, β=1.0).
2. **Ablation** — repeated with β=0.5 to test whether KL regularisation
   was suppressing reconstruction quality. Result: negligible change,
   showing the blur was a capacity limitation from the 2D bottleneck, not
   a regularisation-strength problem.
3. **16D latent** — informed by that finding, re-ran with a 16-dimensional
   latent space. Since a decodable grid isn't possible in 16D, visualised
   instead via a UMAP projection of encoded real images and direct latent
   interpolation between real image pairs.

**Results:**

| Run | Latent dim | β | Final recon loss | Final KL |
|---|---|---|---|---|
| baseline | 2 | 1.0 | 4163.67 | 6.33 |
| beta_low | 2 | 0.5 | 4167.94 | 7.33 |
| latent16 | 16 | 1.0 | 4063.08 | 23.09 |

The 16D latent space and UMAP scatter revealed the latent space is locally
smooth (adjacent slices from the same patient cluster together) but
globally organised by patient identity (inter-patient anatomical variation
dominates over slice-position variation).

**Key files:** `train_vae.py` (dataset, encoder/decoder, loss, training
loop, all three visualisation modes), `job_vae.sh` (Slurm submission,
accepts `--run_name`, `--beta`, `--latent_dim`).

**Results:** see `task1_vae/results/` — `manifold.png`, `reconstructions.png`,
`loss_curve.png` (2D baseline); `umap_scatter.png`, `latent_interpolation.png`,
`reconstructions_16d.png` (16D run).

---

## Section 4.4, Task 2: UNet Segmentation

**Goal:** segment OASIS brain slices into 4 classes, achieving >0.9 DSC on
every class individually, with categorical output and a live inference
demonstration.

**Approach:** standard 4-downsampling-stage UNet (DoubleConv blocks,
32→64→128→256→512 channels, skip connections) trained at native 256x256
resolution (no downsampling, unlike the VAE — segmentation has a strict
accuracy bar that resolution loss would directly threaten). Loss is
combined Dice + cross-entropy, since Dice directly optimises the same
overlap quantity as the DSC evaluation metric.

**Results (final epoch, validation set):**

| Class | DSC | Threshold |
|---|---|---|
| background | 0.9995 | PASS (>0.9) |
| CSF | 0.9535 | PASS (>0.9) |
| grey_matter | 0.9634 | PASS (>0.9) |
| white_matter | 0.9777 | PASS (>0.9) |

Confirmed on a genuine held-out live inference run (`case_453_slice_8`):
background 0.9989, CSF 0.9554, grey_matter 0.9418, white_matter 0.9595 —
all passing, consistent with training-time numbers.

**Key files:** `train_unet.py` (dataset with case/seg filename pairing,
UNet model, Dice+CE loss, per-class DSC tracking), `infer_unet.py`
(standalone fast inference script for the live demo), `job_unet.sh` /
`job_unet_infer.sh`.

**Results:** see `task2_unet/results/` — `loss_curve.png`, `dsc_curve.png`,
`predictions.png`, `live_inference.png`.

---

## Section 4.4, Task 3 (Hard tier): WGAN-GP Brain Generation

**Goal:** generate realistic, diverse brain MRI slices from noise, with
mode collapse fully resolved.

**Approach:** WGAN-GP (Wasserstein GAN with Gradient Penalty), chosen over
a simpler DCGAN specifically for training stability. Prototyped on MNIST
first (Phase A) to validate the training loop mechanics cheaply before
touching OASIS, then ported to OASIS at 64x64 (Phase B).

**Notable finding:** a genuine bug was found and fixed in the sample-grid
visualisation code (an indexing error was rendering only the top row of
each generated image, broadcast into vertical stripes) — this had made
100 epochs of real training look like it produced no structure at all.
After fixing the visualisation and regenerating from the existing
checkpoint (no retraining needed, since the trained weights were never
wrong), the results showed clearly realistic, diverse brain anatomy.

**Results:** 100 epochs (~15,100 generator updates) on the full OASIS
training set. Final random-noise sample grid shows genuine diversity
across ventricle shape, brain outline, and brightness — no mode collapse.

**Key files:** `train_gan_mnist.py` (Phase A prototype), `train_gan_oasis.py`
(Phase B, with checkpoint/resume support), `regenerate_gan_samples.py`
(post-hoc corrected visualisation from a saved checkpoint).

**Results:** see `task3_gan/results/` — `loss_curves.png`,
`CORRECTED_fixed_epoch_100.png`, `CORRECTED_random_samples.png`.

---

## AI Usage

Claude (Anthropic) was used throughout Section 4.4 for architecture design
discussion, debugging (including catching the GAN visualisation bug and an
argparse import-order bug in the UNet inference script), and code
generation, with meaningful back-and-forth on design tradeoffs (resolution,
latent dimensionality, loss function choice) rather than single-prompt
generation. Prompt history available on request per the fair AI use
policy.

## TODO

- [ ] Add Part 1 (DFT/Fourier reconstruction, naive vs FFT timing) section + folder
- [ ] Add Part 2 (Eigenfaces PCA + Random Forest) section + folder
- [ ] Add Part 3.1 (CNN classifier on LFW) section + folder
- [ ] Add Part 3.2 (DAWNBench ResNet-18 on CIFAR10) section + folder
- [ ] Add Advanced Git course completion evidence
- [ ] Review final commit history before submission deadline