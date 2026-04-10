"""Extract raw CIFAR-10 pixels (no ResNet) for tabula rasa vision testing.

Format: same as features but with raw pixels as CHW float arrays.
"""
import torchvision
import torchvision.transforms as T
import struct
import os

transform = T.Compose([
    T.ToTensor(),  # [0,1] range, CHW format
])

train = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=True, download=True, transform=transform)
test = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=False, download=True, transform=transform)

for name, dataset, path in [("train", train, "cifar10_train_pixels.feat"), ("test", test, "cifar10_test_pixels.feat")]:
    n = len(dataset)
    dim = 3 * 32 * 32  # 3072
    print(f"{name}: {n} images, {dim} dims")

    with open(path, 'wb') as f:
        f.write(b'FEAT')
        f.write(struct.pack('<III', n, dim, 10))
        for i in range(n):
            img, label = dataset[i]
            pixels = img.flatten().tolist()  # CHW format, [0,1]
            f.write(struct.pack('<I', label))
            f.write(struct.pack(f'<{dim}f', *pixels))
            if i % 10000 == 0:
                print(f"  {i}/{n}")

    print(f"  Saved: {path} ({os.path.getsize(path):,} bytes)")
