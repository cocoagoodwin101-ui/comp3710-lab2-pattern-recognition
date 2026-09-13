import torch
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

T = 1.0
f0 = 1

# ---- Square wave + Fourier reconstruction (PyTorch) ----
def square_wave_torch(t):
    return torch.sign(torch.sin(2.0 * torch.pi * f0 * t))

def square_wave_fourier_torch(t, f0, Nh):
    n = torch.arange(1, 2 * Nh, 2, device=t.device, dtype=t.dtype)  # odd harmonics 1,3,5...
    terms = torch.sin(2 * torch.pi * n.unsqueeze(1) * f0 * t.unsqueeze(0)) / n.unsqueeze(1)
    return (4 / torch.pi) * terms.sum(dim=0)

# ---- Naive DFT as a GPU matrix multiply ----
def naive_dft_torch(x):
    N = x.shape[0]
    n = torch.arange(N, device=x.device, dtype=torch.float64)
    k = n.reshape(-1, 1)
    angle = -2j * torch.pi * k * n / N
    W = torch.exp(angle.to(torch.complex128))
    x_complex = x.to(torch.complex128)
    return W @ x_complex

# ---- Quick visual check: reconstruct and plot the square wave ----
N_plot = 2048
t_plot = torch.linspace(0.0, T, N_plot, device=device)
square = square_wave_torch(t_plot)

fig = plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(t_plot.cpu().numpy(), square.cpu().numpy(), 'k')
plt.title("Original Square Wave")
plt.ylim(-1.5, 1.5)
plt.grid(True)

for i, Nh in enumerate([1, 3, 5], start=2):
    y = square_wave_fourier_torch(t_plot, f0, Nh)
    plt.subplot(2, 2, i)
    plt.plot(t_plot.cpu().numpy(), y.cpu().numpy(), label=f"N={Nh} harmonics")
    plt.plot(t_plot.cpu().numpy(), square.cpu().numpy(), 'k--', alpha=0.5, label="Square wave")
    plt.title(f"Fourier Approximation, N={Nh}")
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig("square_wave_reconstruction.png")
print("Saved square_wave_reconstruction.png")

# ---- Additional demonstration: higher-order harmonics (N=20, 50) ----
fig2 = plt.figure(figsize=(14, 5))
for i, Nh in enumerate([20, 50], start=1):
    y = square_wave_fourier_torch(t_plot, f0, Nh)
    plt.subplot(1, 2, i)
    plt.plot(t_plot.cpu().numpy(), y.cpu().numpy(), label=f"N={Nh} harmonics")
    plt.plot(t_plot.cpu().numpy(), square.cpu().numpy(), 'k--', alpha=0.5, label="Square wave")
    plt.title(f"Fourier Approximation, N={Nh}")
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.savefig("square_wave_higher_harmonics.png")
print("Saved square_wave_higher_harmonics.png")

# ---- Decompose the constructed square wave back into harmonics via DFT ----
N_dft = 2048
t_dft = (torch.arange(N_dft, device=device, dtype=torch.float64) / N_dft) * T
signal_50 = square_wave_fourier_torch(t_dft, f0, 50)

fft_result = torch.fft.fft(signal_50.to(torch.complex128))
freqs = torch.fft.fftfreq(N_dft, d=(T / N_dft))
magnitude = (2.0 / N_dft) * torch.abs(fft_result)

half = N_dft // 2
freqs_pos = freqs[:half].cpu().numpy()
magnitude_pos = magnitude[:half].cpu().numpy()

plt.figure(figsize=(12, 5))
plt.stem(freqs_pos, magnitude_pos, basefmt=" ")
plt.title("DFT of the Constructed Square Wave (50 harmonics) - Magnitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 20)
plt.grid(True)
for k in range(1, 20, 2):
    plt.axvline(k, color='r', linestyle='--', alpha=0.4)
plt.savefig("dft_decomposition_verification.png")
print("Saved dft_decomposition_verification.png")

# ---- Timing comparison: CPU naive DFT vs GPU naive DFT vs NumPy FFT ----
def naive_dft_numpy(x):
    N = len(x)
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        for n in range(N):
            angle = -2j * np.pi * k * n / N
            X[k] += x[n] * np.exp(angle)
    return X

print("\n--- Timing comparison ---")
for Nsz in [256, 1024, 4096, 8192, 16384, 32768]:
    tt = torch.linspace(0.0, T, Nsz, device=device)
    sig = square_wave_fourier_torch(tt, f0, 50)
    sig_np = sig.cpu().numpy()

    # GPU naive DFT
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    X_gpu = naive_dft_torch(sig)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()

    # NumPy FFT
    t2 = time.time()
    X_fft = np.fft.fft(sig_np)
    t3 = time.time()

    # CPU naive DFT (only for smaller N, it's very slow)
    if Nsz <= 1024:
        t4 = time.time()
        X_cpu = naive_dft_numpy(sig_np)
        t5 = time.time()
        cpu_time = f"{t5 - t4:.6f}s"
    else:
        cpu_time = "skipped (too slow)"

    print(f"N={Nsz:5d}  GPU naive DFT: {t1 - t0:.6f}s   NumPy FFT: {t3 - t2:.6f}s   CPU naive DFT: {cpu_time}")

print("\nDone.")