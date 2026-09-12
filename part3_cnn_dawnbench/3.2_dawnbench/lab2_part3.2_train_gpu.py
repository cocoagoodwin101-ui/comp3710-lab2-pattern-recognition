import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.cuda.amp import autocast, GradScaler
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
print("Using device:", device)

mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)

print("Loading CIFAR-10 directly onto GPU...")
raw_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
raw_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# .data is a numpy uint8 array [N, 32, 32, 3]; skip PIL/transforms entirely
train_images = torch.from_numpy(raw_train.data).float().permute(0, 3, 1, 2) / 255.0
train_labels = torch.tensor(raw_train.targets, dtype=torch.long)
test_images = torch.from_numpy(raw_test.data).float().permute(0, 3, 1, 2) / 255.0
test_labels = torch.tensor(raw_test.targets, dtype=torch.long)

train_images = ((train_images - mean.cpu()) / std.cpu()).to(device).to(memory_format=torch.channels_last)
train_labels = train_labels.to(device)
test_images = ((test_images - mean.cpu()) / std.cpu()).to(device).to(memory_format=torch.channels_last)
test_labels = test_labels.to(device)

print(f"Train tensor on GPU: {train_images.shape}, {train_images.element_size()*train_images.nelement()/1e6:.1f} MB")

# ---- GPU-side augmentation (replaces CPU transforms.Compose) ----
def gpu_random_crop(images, padding=4):
    B, C, H, W = images.shape
    padded = F.pad(images, (padding, padding, padding, padding), mode='reflect')
    max_off = 2 * padding
    off_h = torch.randint(0, max_off + 1, (B,), device=images.device)
    off_w = torch.randint(0, max_off + 1, (B,), device=images.device)
    idx_h = (off_h.view(B, 1, 1, 1) + torch.arange(H, device=images.device).view(1, 1, H, 1)).expand(B, C, H, W)
    idx_w = (off_w.view(B, 1, 1, 1) + torch.arange(W, device=images.device).view(1, 1, 1, W)).expand(B, C, H, W)
    b_idx = torch.arange(B, device=images.device).view(B, 1, 1, 1).expand(B, C, H, W)
    c_idx = torch.arange(C, device=images.device).view(1, C, 1, 1).expand(B, C, H, W)
    return padded[b_idx, c_idx, idx_h, idx_w]

def gpu_random_hflip(images, p=0.5):
    mask = torch.rand(images.shape[0], device=images.device) < p
    out = images.clone()
    out[mask] = torch.flip(images[mask], dims=[3])
    return out

# ---- ResNet-18, CIFAR-adapted (unchanged) ----
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
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
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)

n_train = train_images.shape[0]
steps_per_epoch = n_train // batch_size
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.4, total_steps=epochs * steps_per_epoch,
    pct_start=0.3, div_factor=10, final_div_factor=100
)
scaler = GradScaler()

def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, test_images.shape[0], 1024):
            images = test_images[i:i+1024]
            labels = test_labels[i:i+1024]
            with autocast():
                outputs = model(images)
                outputs_flipped = model(torch.flip(images, dims=[3]))
                probs = F.softmax(outputs, dim=1) + F.softmax(outputs_flipped, dim=1)
            _, predicted = probs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total

print("\nStarting training...")
start_time = time.time()

for epoch in range(epochs):
    model.train()
    perm = torch.randperm(n_train, device=device)
    running_loss = 0.0

    for i in range(0, n_train - batch_size + 1, batch_size):
        idx = perm[i:i+batch_size]
        images = train_images[idx]
        labels = train_labels[idx]

        images = gpu_random_crop(images, padding=4)
        images = gpu_random_hflip(images, p=0.5)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * images.size(0)

    avg_loss = running_loss / n_train
    elapsed = time.time() - start_time

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

torch.save(model.state_dict(), "resnet18_cifar10.pth")
print("Saved resnet18_cifar10.pth")
print("Done.")