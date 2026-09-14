import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.cuda.amp import autocast, GradScaler
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Lets cuDNN benchmark several convolution algorithms on the first batch
# and pick the fastest one for this exact input shape. Safe here because
# every input is always the same size (32x32) - if input sizes varied,
# this would waste time re-benchmarking on every new shape.
torch.backends.cudnn.benchmark = True
print("Using device:", device)

# Standard, well-known per-channel mean/std for CIFAR-10, used to normalise
# pixel values to roughly zero-mean/unit-variance, which helps training
# stability. Kept as GPU tensors (not plain floats) so they broadcast
# directly against image tensors without extra conversion.
mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)

print("Loading CIFAR-10 directly onto GPU...")
# download=True still uses torchvision's normal dataset download/caching
# logic, but we only use it to get the raw files onto disk - we bypass
# its Dataset/DataLoader/transform machinery entirely from here on.
raw_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
raw_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# .data is the raw NumPy array underneath the dataset object, shape
# [N, 32, 32, 3] (image convention: height, width, channels-last).
# permute(0,3,1,2) reorders this to PyTorch's expected [N, C, H, W].
# Dividing by 255.0 converts uint8 pixel values (0-255) to floats in [0,1].
train_images = torch.from_numpy(raw_train.data).float().permute(0, 3, 1, 2) / 255.0
train_labels = torch.tensor(raw_train.targets, dtype=torch.long)
test_images = torch.from_numpy(raw_test.data).float().permute(0, 3, 1, 2) / 255.0
test_labels = torch.tensor(raw_test.targets, dtype=torch.long)

# THE KEY LINE: normalise once, then move the ENTIRE dataset onto the GPU
# as a single tensor, permanently, before training starts 
# so every future access is just GPU-local tensor indexing.
# channels_last reorders the tensor's memory layout to match what the
# A100's Tensor Cores are optimised for during mixed-precision matmuls
train_images = ((train_images - mean.cpu()) / std.cpu()).to(device).to(memory_format=torch.channels_last)
train_labels = train_labels.to(device)
test_images = ((test_images - mean.cpu()) / std.cpu()).to(device).to(memory_format=torch.channels_last)
test_labels = test_labels.to(device)

print(f"Train tensor on GPU: {train_images.shape}, {train_images.element_size()*train_images.nelement()/1e6:.1f} MB")

# ---- GPU-side augmentation (replaces CPU torchvision.transforms.Compose) ----
# Same augmentations as before (random crop with padding, random flip),
# just re-implemented as batched GPU tensor operations instead of
# per-image CPU/PIL operations, so no CPU involvement and no data ever
# leaves the GPU during training.
def gpu_random_crop(images, padding=4):
    B, C, H, W = images.shape
    # Reflect-pad the image so a crop near the edge doesn't introduce
    # artificial black borders - "reflect" mirrors nearby pixels instead.
    padded = F.pad(images, (padding, padding, padding, padding), mode='reflect')
    max_off = 2 * padding
    # A different random crop offset per image in the batch (not one
    # shared offset for the whole batch), matching what
    # transforms.RandomCrop would have done per-image on CPU.
    off_h = torch.randint(0, max_off + 1, (B,), device=images.device)
    off_w = torch.randint(0, max_off + 1, (B,), device=images.device)
    # Build an index grid per image that selects the right HxW window out
    # of the padded image, then gather it all in one batched indexing
    # operation - no explicit Python loop over individual images.
    idx_h = (off_h.view(B, 1, 1, 1) + torch.arange(H, device=images.device).view(1, 1, H, 1)).expand(B, C, H, W)
    idx_w = (off_w.view(B, 1, 1, 1) + torch.arange(W, device=images.device).view(1, 1, 1, W)).expand(B, C, H, W)
    b_idx = torch.arange(B, device=images.device).view(B, 1, 1, 1).expand(B, C, H, W)
    c_idx = torch.arange(C, device=images.device).view(1, C, 1, 1).expand(B, C, H, W)
    return padded[b_idx, c_idx, idx_h, idx_w]

def gpu_random_hflip(images, p=0.5):
    # A random boolean mask decides which images in the batch get
    # flipped this step; only those images are flipped, the rest pass
    # through unchanged - equivalent to RandomHorizontalFlip(p=0.5) but
    # applied to a whole batch of GPU tensors at once.
    mask = torch.rand(images.shape[0], device=images.device) < p
    out = images.clone()
    out[mask] = torch.flip(images[mask], dims=[3])
    return out

# ---- ResNet-18, CIFAR-adapted ----
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # Identity shortcut by default. Only needs an actual conv when the
        # block changes channel count or spatial size (stride!=1) - the
        # shortcut's only job then is reshaping x to match out's shape so
        # the addition below is even valid, not extracting new features.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # The residual/skip connection: add the (possibly reshaped) input
        # back onto the output of the two convolutions. This is what gives
        # gradients a direct path backward during backprop, avoiding the
        # vanishing-gradient problem in networks this deep.
        out += self.shortcut(x)
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        # CIFAR-adapted stem: a gentle 3x3 stride-1 conv with NO maxpool,
        # instead of the standard ImageNet ResNet-18's 7x7 stride-2 conv +
        # maxpool. That standard stem would shrink a 32x32 CIFAR image 4x
        # almost immediately, destroying spatial detail before the network
        # even starts - this version preserves resolution instead.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Four stages, each built from 2 BasicBlocks. layer1 keeps spatial
        # size the same (stride=1); layer2-4 each downsample by 2x on
        # their first block while doubling the channel count.
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        # Collapses whatever spatial size remains down to 1x1 per channel,
        # giving a fixed-length 512-value vector regardless of input size.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.flatten(1)
        return self.fc(out)

model = ResNet18().to(device).to(memory_format=torch.channels_last)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

epochs = 30
batch_size = 512
# label_smoothing softens the training targets slightly (instead of a
# hard 1.0 for the correct class), discouraging the model from becoming
# overconfident - a small, cheap regularisation trick.
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# SGD + Nesterov momentum, chosen over Adam because it empirically reaches
# a higher final accuracy on CNN image-classification tasks, even though
# it needs more careful learning-rate tuning to get there.
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)

n_train = train_images.shape[0]
steps_per_epoch = n_train // batch_size
# OneCycleLR: learning rate rises to max_lr then anneals back down over
# training. The rising phase lets the network explore quickly early on;
# the falling phase lets it settle precisely by the end. This schedule
# (not just the optimizer choice) is the actual mechanism behind
# fast-training recipes reaching high accuracy in a short time budget.
# max_lr=0.4 was found by direct experimentation - both 0.35 and 0.45/0.5
# were tested and performed worse at the same epoch count/batch size.
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.4, total_steps=epochs * steps_per_epoch,
    pct_start=0.3, div_factor=10, final_div_factor=100
)
# Handles mixed-precision training safely: FP16 has a narrow numeric
# range, so gradients can underflow to zero during backprop. GradScaler
# scales the loss up before backward() to keep gradients representable,
# then unscales before the optimizer actually updates the weights.
scaler = GradScaler()

def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        # Iterate over the test set in fixed-size chunks manually (no
        # DataLoader needed here either, since test_images already lives
        # entirely on the GPU).
        for i in range(0, test_images.shape[0], 1024):
            images = test_images[i:i+1024]
            labels = test_labels[i:i+1024]
            with autocast():
                outputs = model(images)
                # ---- Test-Time Augmentation (TTA) ----
                # Run the SAME batch through the model a second time,
                # horizontally flipped. A flipped object is still the same
                # object, so this gives the model two independent "looks"
                # at each test image. Summing the two softmax probability
                # distributions averages out prediction noise between the
                # two views, giving a small, essentially free accuracy
                # boost - this only happens at evaluation time, never
                # during training, so it costs no extra training time at
                # all, just a second forward pass per evaluation.
                outputs_flipped = model(torch.flip(images, dims=[3]))
                probs = F.softmax(outputs, dim=1) + F.softmax(outputs_flipped, dim=1)
            # Highest combined probability wins - .max(1) returns the
            # value and index of the largest score per row; we only need
            # the index (the predicted class).
            _, predicted = probs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total

print("\nStarting training...")
start_time = time.time()

for epoch in range(epochs):
    model.train()
    # Shuffle indices ON the GPU (device=device) rather than shuffling a
    # CPU-side dataset - keeps the whole shuffling step off the CPU too.
    perm = torch.randperm(n_train, device=device)
    running_loss = 0.0

    for i in range(0, n_train - batch_size + 1, batch_size):
        # Slicing into a tensor that's already resident on the GPU -
        # this replaces the entire DataLoader iteration step, and it's
        # essentially instantaneous compared to the old pipeline's
        # equivalent (disk read + CPU augment + transfer) per batch.
        idx = perm[i:i+batch_size]
        images = train_images[idx]
        labels = train_labels[idx]

        # GPU-side augmentation applied fresh every batch, same as the
        # old CPU pipeline's per-batch augmentation, just relocated.
        images = gpu_random_crop(images, padding=4)
        images = gpu_random_hflip(images, p=0.5)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # The mixed-precision training step: scale loss -> backward pass
        # -> unscale and step the optimizer -> update the scale factor for
        # next time -> advance the OneCycle schedule by one step.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * images.size(0)

    avg_loss = running_loss / n_train
    elapsed = time.time() - start_time

    # Only actually run the (relatively expensive, TTA-doubled) evaluation
    # every 5 epochs plus the final one, rather than every single epoch -
    # this reclaims real training time that would otherwise be spent on
    # printouts rather than learning.
    if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
        test_acc = evaluate()
        print(f"Epoch {epoch+1}/{epochs}  loss: {avg_loss:.4f}  test_acc: {test_acc:.4f}  elapsed: {elapsed:.1f}s")
        if test_acc >= 0.94:
            print(f"\n>>> Hit 94% accuracy at epoch {epoch+1}, elapsed {elapsed:.1f}s <<<")
    else:
        print(f"Epoch {epoch+1}/{epochs}  loss: {avg_loss:.4f}  elapsed: {elapsed:.1f}s")

total_time = time.time() - start_time
final_acc = evaluate()
print(f"\nFinal test accuracy: {final_acc:.4f}")
print(f"Total training time: {total_time:.1f}s")

# Saved weights are what lab2_part3.2_demo.py loads for the live
# one-epoch-plus-inference demonstration - this file is the one that
# must be re-run last if you want the saved weights to match whatever
# numbers you're about to quote in the demo.
torch.save(model.state_dict(), "resnet18_cifar10.pth")
print("Saved resnet18_cifar10.pth")
print("Done.")