"""A function to load the Fashion MNIST digit dataset."""

from typing import Tuple

import torchvision
import torchvision.transforms as transforms


def load_fmnist(data_root, download) -> Tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    """Load Fashion MNIST (training and test set)."""
    
    # Define the transform for the data.
    transform = transforms.Compose([
        # torchvision.transforms.ToTensor(),    
        torchvision.transforms.Resize(size=(32,32), antialias=None),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    # Initialize Datasets. MNIST will automatically download if not present
    trainset = torchvision.datasets.FashionMNIST(
        root=data_root, train=True, download=download, transform=transform
    )
    testset = torchvision.datasets.FashionMNIST(
        root=data_root, train=False, download=download, transform=transform
    )

    # Return the datasets
    return trainset, testset
