# COMP3710 Pattern Analysis - Lab 2: Pattern Recognition

Lab Demonstration 2 submission. This lab covers dimensionality reduction,
classification, and deep learning pipelines using PyTorch on UQ's Rangpur
HPC cluster (comp3710 partition, A100 GPUs).

## Repository Structure

```
Lab2/
├── README.md
├── .gitignore
├── demo2_q1-3_overview.pdf   # teaching-staff reference doc, Parts 1-3
├── part1_dft/                # Part 1
│   ├── comp3710_lab2.py
│   ├── job.sh
│   ├── lab2.txt
│   └── results/
├── part2_eigenfaces/         # Part 2
│   ├── lab2_part2_eigenfaces.py
│   ├── job2.sh
│   ├── lab2_part2.txt
│   └── results/
├── part3_cnn_dawnbench/      # Part 3
│   ├── 3.1_cnn_classifier/
│   │   ├── lab2_part3.1_cnn.py
│   │   ├── job3.sh
│   │   ├── lab2_part3.1.txt
│   │   └── results/
│   └── 3.2_dawnbench/
│       ├── lab2_part3.2_train.py      # DataLoader baseline
│       ├── lab2_part3.2_train_gpu.py  # final GPU-resident pipeline
│       ├── lab2_part3.2_demo.py       # live epoch + inference script
│       ├── job4.sh, job5.sh, job6.sh
│       ├── lab2_part3.2.txt
├── task1_vae/                # Section 4.4, Task 1
│   ├── train_vae.py
│   ├── job_vae.sh
│   ├── explanation.txt
│   └── results/
├── task2_unet/                # Section 4.4, Task 2
│   ├── train_unet.py
│   ├── infer_unet.py
│   ├── job_unet.sh
│   ├── job_unet_infer.sh
│   ├── lab2_part4_task2.txt
│   └── results/
└── task3_gan/                  # Section 4.4, Task 3 (Hard tier)
    ├── train_gan_mnist.py    # Phase A: MNIST prototype
    ├── job_gan_mnist.sh
    ├── train_gan_oasis.py    # Phase B: OASIS training
    ├── job_gan_oasis.sh
    ├── regenerate_gan_samples.py
    ├── job_gan_regen.sh
    ├── lab2_part4_task3.txt
    └── results/
```

## Environment

- **Cluster:** Rangpur HPC, `comp3710` partition (A100 GPUs), jobs submitted
  via `sbatch`.
- **Framework:** PyTorch, conda environment `torch` (Python 3.11).
- **Dataset:** OASIS preprocessed brain MRI slices, `/home/groups/comp3710/OASIS`
  (256x256 grayscale PNGs; segmentation masks with 4 classes: background,
  CSF, grey matter, white matter). CIFAR-10 and Labeled Faces in the Wild
  (LFW) downloaded via torchvision/scikit-learn for Parts 2/3.

---

## Part 1: Discrete Fourier Transform

**Goal:** reconstruct a square wave from its Fourier series, port the
implementation from NumPy to PyTorch, and compare a GPU-native naive DFT
against NumPy's FFT across increasing signal sizes.

**Approach:** the square wave and its Fourier series reconstruction were
vectorised in PyTorch (an outer product across harmonics × time samples,
rather than a Python loop) so the whole computation can be dispatched to
the GPU in one shot. The naive DFT was expressed as multiplication by an
explicit N×N complex exponential matrix (`W @ x`), turning an O(N²) nested
loop into a single dense matmul — the form GPUs are actually built to
accelerate. This was benchmarked against NumPy's O(N log N) FFT and a
brute-force CPU-loop DFT across N = 256 up to 32,768.

**Results:**

| N | GPU naive DFT | NumPy FFT | Faster |
|---|---|---|---|
| 256 | 0.064s | 0.0047s | NumPy FFT |
| 1,024 | 0.00065s | 0.000056s | NumPy FFT |
| 4,096 | 0.0015s | 0.00021s | NumPy FFT |
| 8,192 | 0.0049s | 0.00029s | NumPy FFT |
| 16,384 | 0.019s | 0.00053s | NumPy FFT |
| 32,768 | 0.462s | 0.0013s | NumPy FFT |

NumPy FFT wins at every size tested, and the gap *widens* with N — direct
evidence that a smarter algorithm (O(N log N)) beats a brute-force one
(O(N²)) parallelised on a GPU, until N is large enough for raw parallelism
to outweigh doing quadratically more total work. The reconstruction plot
also clearly shows the Gibbs phenomenon: adding more harmonics sharpens
the approximation but never removes the fixed-height overshoot at the
signal's discontinuities, just compresses it closer to the edge.

Two further demonstrations were added to directly answer the lab
sheet's remaining questions:

- **Higher-order harmonics (N=20, N=50):** away from the
  discontinuities, accuracy improves substantially as harmonic count
  increases — the flat regions converge tightly to ±1.0. At the
  discontinuities themselves, the Gibbs overshoot persists at roughly
  the same fixed height (~1.18–1.20) regardless of N, only compressing
  in width — confirming more harmonics improve sharpness everywhere
  except exactly at the jumps, where the overshoot is a genuine,
  unavoidable property of finite Fourier series approximating a
  discontinuous function.
- **DFT decomposition verification:** the constructed 50-harmonic
  square wave was passed through `torch.fft.fft` and its magnitude
  spectrum plotted. The result shows sharp peaks exactly at every odd
  harmonic (1, 3, 5, ... 19 Hz) with magnitude decaying ~1/n, and
  effectively zero magnitude at even harmonics — confirming the DFT
  correctly recovers precisely the frequency components used to
  construct the signal, verifying the forward (series → signal) and
  inverse (signal → spectrum) operations are consistent.

**Key files:** `comp3710_lab2.py` (square wave, Fourier reconstruction,
higher-harmonic demonstration, DFT decomposition verification, naive
DFT, timing sweep), `job.sh`.

**Results:** see `part1_dft/results/` — `square_wave_reconstruction.png`,
`square_wave_higher_harmonics.png`, `dft_decomposition_verification.png`.

---

## Part 2: Eigenfaces (PCA + Random Forest)

**Goal:** compute Eigenfaces via PCA on the Labeled Faces in the Wild
(LFW) dataset and classify identities with a Random Forest baseline.

**Approach:** PCA was computed via SVD directly on the mean-centred data
matrix (`np.linalg.svd`) rather than eigendecomposing an explicit
covariance matrix — numerically equivalent, but more stable and avoids
ever forming the covariance matrix explicitly. The top 150 components
were used to project both train and test sets into "face space," and a
Random Forest (150 estimators) was trained on the projected features.

**Results:**

- Dataset: 1,288 samples, 1,850 raw pixel features, 7 identity classes.
- PCA: 150 components retained ~95% cumulative explained variance (see
  compactness curve).
- Random Forest test accuracy: **62.4%** overall.

| Class | Precision | Recall | Support |
|---|---|---|---|
| Ariel Sharon | 0.00 | 0.00 | 13 |
| Colin Powell | 0.69 | 0.60 | 60 |
| Donald Rumsfeld | 0.62 | 0.19 | 27 |
| George W Bush | 0.62 | 0.90 | 146 |
| Gerhard Schroeder | 0.53 | 0.32 | 25 |
| Hugo Chavez | 0.57 | 0.53 | 15 |
| Tony Blair | 0.63 | 0.33 | 44 |

The low overall accuracy is driven by significant class imbalance (George
W Bush alone is ~45% of the test set) rather than a modelling error — the
classifier over-predicts the majority class, and this is directly
addressed by Part 3.1's CNN, which substantially outperforms this
PCA+Random Forest baseline on the identical dataset.

**Key files:** `lab2_part2_eigenfaces.py`, `job2.sh`.

**Results:** see `part2_eigenfaces/results/` — `eigenfaces_gallery.png`,
`compactness.png`.

---

## Part 3.1: CNN Classifier (LFW)

**Goal:** implement a simple CNN (two 3×3 conv layers, 32 filters each,
dense classifier head) for the same LFW dataset used in Part 2, and
compare against the PCA+Random Forest baseline.

**Approach:** raw LFW images (`[N, H, W]`) were reshaped to add a channel
dimension (`[N, 1, H, W]`) for `Conv2d`. Trained with Adam and
cross-entropy loss over 30 epochs.

**Results:** **88.2-88.5% test accuracy**, a large improvement over Part
2's 62.4% PCA+Random Forest baseline. This demonstrates the core
advantage of CNNs over a linear-decomposition-plus-classifier pipeline:
convolution preserves 2D spatial locality (which pixels are near which),
and the feature extractor and classifier are trained jointly end-to-end,
whereas PCA's components are optimised purely for variance, with no
awareness of which variance actually helps distinguish classes.

**Key files:** `lab2_part3.1_cnn.py`, `job3.sh`.

**Results:** see `3.1_cnn_classifier/results/` — `cnn_loss_curve.png`.

---

## Part 3.2: DAWNBench Challenge (ResNet-18 on CIFAR-10)

**Goal:** implement ResNet-18 from scratch (no pretrained weights) for
CIFAR-10, hit >90% accuracy in under 30 minutes, and chase the harder
stretch goal of ~94% accuracy in a time equivalent to or faster than the
~360-second V100 GPU reference benchmark.

**Approach:** extensive, deliberately isolated experimentation (full
iteration-by-iteration log in `Part3.2_DAWNBench_Story.txt`). Key
techniques: a CIFAR-adapted ResNet-18 stem (3×3 stride-1 conv, no
maxpool, since the standard ImageNet stem destroys too much spatial
detail on 32×32 inputs), mixed precision via `torch.cuda.amp` with
`GradScaler`, SGD + Nesterov momentum + OneCycle LR (chosen over Adam —
empirically reaches higher final accuracy on CNN vision tasks), and
Test-Time Augmentation (horizontal-flip-averaged predictions at eval
time, near-zero cost). The single biggest discovery: the original
`DataLoader`-based pipeline was CPU-bound, not GPU-bound — replacing it
with the entire dataset preloaded once onto the A100 as a single GPU
tensor, with augmentation (random crop, flip) rewritten as GPU tensor
ops, gave a ~7x per-epoch speedup.

**Results (key milestones):**

| Stage | Configuration | Accuracy | Time |
|---|---|---|---|
| Baseline | 20ep, DataLoader pipeline | 93.06% | 350.3s |
| Best under 360s (DataLoader) | 22ep, TTA, cuDNN+channels_last | 93.86% | 342.8s |
| Confirmed recipe hits 94%+ | 30ep, DataLoader pipeline | 94.36% | 470.5s |
| **Final: GPU-resident pipeline** | **30ep, data preloaded on GPU** | **94.45%** | **64.9s** |

Final result clears the 94% target with real margin and beats the 360s
V100 benchmark by roughly 5.5x, on an A100 (a faster GPU than the
original V100 reference, disclosed for fair comparison). Multiple
controlled experiments (learning rate sweep, batch size 512 vs 1024,
Cutout augmentation with/without) are documented in full in
`Part3.2_DAWNBench_Story.txt`, including a batch-size negative result
(1024 consistently underperformed 512, consistent with the known
"generalization gap" of large-batch training) and an abandoned EMA
(Exponential Moving Average) attempt with a known implementation bug,
reverted rather than debugged further under time constraints.

Mark 2 (live epoch + inference during the demo) is handled by a
deliberately separate script (`lab2_part3.2_demo.py`) that loads the
already-trained weights, runs one real training epoch live and timed,
then performs inference on a small test batch — verified working
(11.4s/epoch, 16/16 correct sample predictions).

**Key files:** `lab2_part3.2_train.py` (DataLoader baseline),
`lab2_part3.2_train_gpu.py` (final GPU-resident pipeline, produces the
saved model weights), `lab2_part3.2_demo.py` (live demo script),
`job4.sh` / `job5.sh` / `job6.sh`.

**Results:** no plot images were generated for this part (results were
tracked via stdout logs/printed metrics rather than saved figures); see
`Part3.2_DAWNBench_Story.txt` for the complete iteration log,
concept explanations, and results table.

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

Claude (Anthropic) was used throughout this lab for architecture design
discussion, debugging, and code generation, with meaningful back-and-forth
on design tradeoffs (e.g. resolution, latent dimensionality, loss function
choice, learning rate schedules, data pipeline design) rather than
single-prompt generation. Notable debugging assists included: an
SBATCH heredoc formatting error, a Slurm partition/account permissions
issue, a GAN visualisation indexing bug, and identifying a CPU-bound
DataLoader bottleneck in the DAWNBench challenge (Part 3.2) that was
resolved with a GPU-resident data pipeline. Prompt history and full
iteration logs (see `part3_cnn_dawnbench/3.2_dawnbench/Part3.2_DAWNBench_Story.txt`)
available on request per the fair AI use policy.

## TODO

- [ ] Add Advanced Git course completion evidence
- [ ] Review final commit history before submission deadline