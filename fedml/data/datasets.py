"""All available dataset definitions."""

from typing import Tuple
import os
import joblib

from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms


def load_cifar10(data_root, download) -> Tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    """Load CIFAR-10 (training and test set)."""
    
    # Define the transform for the data.
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Initialize Datasets. CIFAR-10 will automatically download if not present
    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=download, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=download, transform=test_transform
    )
    
    # Return the datasets
    return trainset, testset


def load_cifar100(data_root, download) -> Tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    """Load CIFAR-100 (training and test set)."""
    
    # Define the transform for the data.
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Initialize Datasets. CIFAR-100 will automatically download if not present
    trainset = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=download, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=download, transform=test_transform
    )
    
    # Return the datasets
    return trainset, testset


def load_emnist(data_root, download, split="mnist") -> Tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    """Load MNIST (training and test set)."""
    
    # Define the transform for the data.
    transform = transforms.Compose([
        # transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.1307,))
    ])
    
    # Initialize Datasets. EMNIST will automatically download if not present
    trainset = torchvision.datasets.EMNIST(
        root=data_root, train=True, split=split, download=download, transform=transform
    )
    testset = torchvision.datasets.EMNIST(
        root=data_root, train=False, split=split, download=download, transform=transform
    )

    # Return the datasets
    return trainset, testset

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

def load_mnist(data_root, download) -> Tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    """Load MNIST (training and test set)."""
    
    # Define the transform for the data.
    transform = transforms.Compose([
        # transforms.ToTensor(),
        torchvision.transforms.Resize(size=(32,32), antialias=None),
        transforms.Normalize((0.1307,), (0.1307,))
    ])

    # Initialize Datasets. MNIST will automatically download if not present
    trainset = torchvision.datasets.MNIST(
        root=data_root, train=True, download=download, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root=data_root, train=False, download=download, transform=transform
    )

    # Return the datasets
    return trainset, testset

def load_stl10(data_root, download) -> Tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    """Load STL-10 (training and test set)."""
    
    # Define the transform for the data.
    transform = transforms.Compose([
        torchvision.transforms.Resize(size=(32,32), antialias=None),
        # torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Initialize Datasets. STL-10 will automatically download if not present
    trainset = torchvision.datasets.STL10(
        root=data_root, split="train", folds=None, download=download, transform=transform
    )
    testset = torchvision.datasets.STL10(
        root=data_root, split="test", folds=None, download=download, transform=transform
    )
    
    # Return the datasets
    return trainset, testset

def load_tinyimagenet(data_root, download) -> Tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    """Load Tiny Imagenet (training and test set)."""
    
    # Define the transform for the data.
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.RandomResizedCrop(64),
        transforms.RandomCrop(64, padding=4),
        # transforms.RandomRotation([-30.0, 30.0]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Initialize Datasets. Tiny Imagenet requires that dataset is already downloaded.
    trainset = torchvision.datasets.ImageFolder(
        root=data_root + '/train', transform=train_transform
    )
    testset= torchvision.datasets.ImageFolder(
        root=data_root + '/val', transform=test_transform
    )
    
    # Return the datasets
    return trainset, testset

def load_ucimlparkinsons(use_lib, data_root=None, download=False) -> Tuple:
    """Load UCI Parkinsons Telemonitoring dataset (training and test set)."""

    # Check if a cached version exists on disk
    if data_root is not None:
        cache_path = os.path.join(data_root, "uciml_parkinsons.joblib")
        if os.path.exists(cache_path):
            # print(f"Loading Parkinsons dataset from cache: {cache_path}")
            X, y = joblib.load(cache_path)
            return X, y

    if use_lib:
        # Load dataset using ucimlrepo library
        from ucimlrepo import fetch_ucirepo
        from sklearn.model_selection import train_test_split

        dataset = fetch_ucirepo(id=189)

        X = dataset.data.features.copy()
        y = dataset.data.targets["motor_UPDRS"].copy()   # regression target
    else:
        raise NotImplementedError("Loading UCI ML Parkinsons dataset without ucimlrepo library is not implemented yet.")

    # Save to disk if data_root is provided
    if data_root is not None:
        os.makedirs(data_root, exist_ok=True)
        joblib.dump((X, y), cache_path)
        # print(f"Dataset cached to: {cache_path}")

    # Return the datasets
    return X, y