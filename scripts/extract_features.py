"""Extract ResNet18 features from CIFAR-10 for CTM training.

Outputs a binary file with format:
  FEAT magic (4 bytes)
  n_samples (u32)
  feature_dim (u32)
  n_classes (u32)
  for each sample:
    class_id (u32)
    features (feature_dim × f32)

Class names: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""

import torch
import torchvision
import torchvision.transforms as T
import struct
import sys

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load ResNet18 pre-trained on ImageNet
    print("Loading ResNet18 backbone...")
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.eval()
    # Remove the classification head — we want features, not ImageNet predictions
    # ResNet18 feature dim = 512
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # remove fc layer
    feature_extractor = feature_extractor.to(device)

    # CIFAR-10 with ResNet-appropriate transforms
    transform = T.Compose([
        T.Resize(224),  # ResNet expects 224x224
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Downloading CIFAR-10...")
    train_set = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=False, download=True, transform=transform)

    classes = train_set.classes
    print(f"Classes: {classes}")

    for split_name, dataset, path in [
        ("train", train_set, "cifar10_train.feat"),
        ("test", test_set, "cifar10_test.feat"),
    ]:
        print(f"\nExtracting {split_name} features ({len(dataset)} images)...")
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(loader):
                images = images.to(device)
                features = feature_extractor(images).squeeze(-1).squeeze(-1)  # [B, 512]
                all_features.append(features.cpu())
                all_labels.append(labels)
                if batch_idx % 50 == 0:
                    print(f"  batch {batch_idx}/{len(loader)}")

        features = torch.cat(all_features, dim=0)  # [N, 512]
        labels = torch.cat(all_labels, dim=0)  # [N]

        n_samples = features.shape[0]
        feature_dim = features.shape[1]
        n_classes = 10

        print(f"  {n_samples} samples, {feature_dim} features, {n_classes} classes")

        # Save as binary
        with open(path, 'wb') as f:
            f.write(b'FEAT')
            f.write(struct.pack('<III', n_samples, feature_dim, n_classes))
            for i in range(n_samples):
                f.write(struct.pack('<I', int(labels[i])))
                f.write(struct.pack(f'<{feature_dim}f', *features[i].tolist()))

        import os
        size = os.path.getsize(path)
        print(f"  Saved to {path} ({size:,} bytes)")

    # Also save class names
    with open("cifar10_classes.txt", "w") as f:
        for i, name in enumerate(classes):
            f.write(f"{i} {name}\n")

    print("\nDone! Files: cifar10_train.feat, cifar10_test.feat, cifar10_classes.txt")

if __name__ == "__main__":
    main()
