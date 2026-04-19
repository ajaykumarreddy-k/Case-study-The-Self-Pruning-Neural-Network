"""
CIFAR-10 Dataset Loaders
========================
Standard CIFAR-10 mean/std normalization values used throughout this project:
  mean=(0.4914, 0.4822, 0.4465)
  std =(0.2023, 0.1994, 0.2010)
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 2,
):
    """
    Return (train_loader, test_loader) for CIFAR-10.

    Args:
        batch_size : Samples per batch (default 128).
        data_dir   : Directory to cache the dataset (default './data').
        num_workers: DataLoader worker processes (default 2).

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    # Standard CIFAR-10 per-channel mean and std
    _mean = (0.4914, 0.4822, 0.4465)
    _std  = (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
