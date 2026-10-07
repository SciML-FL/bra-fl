"""A function to load and split the desired dataset among clients."""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
        
from fedml.data.split import CustomDataset, CustomRegressionDataset, CustomSubset, split_data
import torch
import torchvision.transforms as transforms

def load_data(dataset_name: str, 
              dataset_path: str, 
              dataset_down: bool,
              random_seed: int):

    custom_trainset, custom_testset = None, None

    if dataset_name == "MNIST":
        from .datasets import load_mnist
        trainset, testset = load_mnist(data_root=dataset_path, download=dataset_down)

        # Extract transforms from train and test set.
        tr_transform = transforms.Compose(trainset.transform.transforms) if trainset.transform else None
        tr_target_transform = transforms.Compose(trainset.target_transform.transforms) if trainset.target_transform else None
        ts_transform = transforms.Compose(testset.transform.transforms) if testset.transform else None
        ts_target_transform = transforms.Compose(testset.target_transform.transforms) if testset.target_transform else None

        # Build Custom datasets
        custom_trainset = CustomDataset(data=trainset.data.unsqueeze(1)/255.0, targets=trainset.targets, transform=tr_transform, target_transform=tr_target_transform)
        custom_testset = CustomDataset(data=testset.data.unsqueeze(1)/255.0, targets=testset.targets, transform=ts_transform, target_transform=ts_target_transform)

    elif dataset_name == "EMNIST-DIGITS":
        from .datasets import load_emnist
        trainset, testset = load_emnist(data_root=dataset_path, download=dataset_down, split="digits")

        # Extract transforms from train and test set.
        tr_transform = transforms.Compose(trainset.transform.transforms) if trainset.transform else None
        tr_target_transform = transforms.Compose(trainset.target_transform.transforms) if trainset.target_transform else None
        ts_transform = transforms.Compose(testset.transform.transforms) if testset.transform else None
        ts_target_transform = transforms.Compose(testset.target_transform.transforms) if testset.target_transform else None

        # Build Custom datasets
        custom_trainset = CustomDataset(data=trainset.data.unsqueeze(1)/255.0, targets=trainset.targets, transform=tr_transform, target_transform=tr_target_transform)
        custom_testset = CustomDataset(data=testset.data.unsqueeze(1)/255.0, targets=testset.targets, transform=ts_transform, target_transform=ts_target_transform)

    elif dataset_name == "CIFAR-10":
        from .datasets import load_cifar10
        trainset, testset = load_cifar10(data_root=dataset_path, download=dataset_down)

        # Extract transforms from train and test set.
        tr_transform = transforms.Compose(trainset.transform.transforms) if trainset.transform else None
        tr_target_transform = transforms.Compose(trainset.target_transform.transforms) if trainset.target_transform else None
        ts_transform = transforms.Compose(testset.transform.transforms) if testset.transform else None
        ts_target_transform = transforms.Compose(testset.target_transform.transforms) if testset.target_transform else None

        # Build Custom datasets
        # Modify data to have [S, C, H, W] format
        custom_trainset = CustomDataset(data=trainset.data.transpose((0, 3, 1, 2))/255.0, targets=trainset.targets, transform=tr_transform, target_transform=tr_target_transform)
        custom_testset = CustomDataset(data=testset.data.transpose((0, 3, 1, 2))/255.0, targets=testset.targets, transform=ts_transform, target_transform=ts_target_transform)

    elif dataset_name == "CIFAR-100":
        from .datasets import load_cifar100
        trainset, testset = load_cifar100(data_root=dataset_path, download=dataset_down)

        # Extract transforms from train and test set.
        tr_transform = transforms.Compose(trainset.transform.transforms) if trainset.transform else None
        tr_target_transform = transforms.Compose(trainset.target_transform.transforms) if trainset.target_transform else None
        ts_transform = transforms.Compose(testset.transform.transforms) if testset.transform else None
        ts_target_transform = transforms.Compose(testset.target_transform.transforms) if testset.target_transform else None

        # Build Custom datasets
        # Modify data to have [S, C, H, W] format
        custom_trainset = CustomDataset(data=trainset.data.transpose((0, 3, 1, 2))/255.0, targets=trainset.targets, transform=tr_transform, target_transform=tr_target_transform)
        custom_testset = CustomDataset(data=testset.data.transpose((0, 3, 1, 2))/255.0, targets=testset.targets, transform=ts_transform, target_transform=ts_target_transform)

    elif dataset_name == "FMNIST":
        # Load Fashion-MNIST dataset
        from .datasets import load_fmnist
        trainset, testset = load_fmnist(data_root=dataset_path, download=dataset_down)

        # Extract transforms from train and test set.
        tr_transform = transforms.Compose(trainset.transform.transforms) if trainset.transform else None
        tr_target_transform = transforms.Compose(trainset.target_transform.transforms) if trainset.target_transform else None
        ts_transform = transforms.Compose(testset.transform.transforms) if testset.transform else None
        ts_target_transform = transforms.Compose(testset.target_transform.transforms) if testset.target_transform else None

        # Build Custom datasets
        custom_trainset = CustomDataset(data=trainset.data.unsqueeze(1)/255.0, targets=trainset.targets, transform=tr_transform, target_transform=tr_target_transform)
        custom_testset = CustomDataset(data=testset.data.unsqueeze(1)/255.0, targets=testset.targets, transform=ts_transform, target_transform=ts_target_transform)

    elif dataset_name == "STL-10":
        # Load STL-10 dataset
        from .datasets import load_stl10
        trainset, testset = load_stl10(out_dir=dataset_path, download=dataset_down)

        # Extract transforms from train and test set.
        tr_transform = transforms.Compose(trainset.transform.transforms) if trainset.transform else None
        tr_target_transform = transforms.Compose(trainset.target_transform.transforms) if trainset.target_transform else None
        ts_transform = transforms.Compose(testset.transform.transforms) if testset.transform else None
        ts_target_transform = transforms.Compose(testset.target_transform.transforms) if testset.target_transform else None

        # Build Custom datasets
        # For some weird reason STL-10 has labels instead 
        # of targets adding additional attribute targets 
        # to make it consistent with other datasets
        custom_trainset = CustomDataset(data=trainset.data, targets=trainset.labels, transform=tr_transform, target_transform=tr_target_transform)
        custom_testset = CustomDataset(data=testset.data, targets=testset.labels, transform=ts_transform, target_transform=ts_target_transform)

    elif dataset_name == "TINY-IMAGENET":
        # Load Tiny-Imagenet dataset
        from .datasets import load_tinyimagenet
        trainset, testset = load_tinyimagenet(data_root=dataset_path, download=dataset_down)
        trainset.targets = torch.tensor(trainset.targets)
        testset.targets = torch.tensor(testset.targets)

        custom_trainset = CustomSubset(trainset, indices=list(range(len(trainset))))
        custom_testset= CustomSubset(testset, indices=list(range(len(testset))))

    elif dataset_name == "UCIML-PARKINSONS":
        # Load UCI ML Parkinsons dataset
        from .datasets import load_ucimlparkinsons
        features, targets = load_ucimlparkinsons(use_lib=True, data_root=dataset_path, download=dataset_down)
        
        # Extract train and test splits
        X_train, X_test, y_train, y_test = train_test_split(
            features.values,
            targets.values,
            test_size=0.2,
            random_state=random_seed
        )

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Build Custom datasets
        custom_trainset = CustomRegressionDataset(data=X_train, targets=y_train)
        custom_testset = CustomRegressionDataset(data=X_test, targets=y_test)

    else:
        raise ValueError(f"Invalid dataset {dataset_name} requested.")

    # Performing cleanups
    # del trainset, testset

    return custom_trainset, custom_testset

def load_and_fetch_split(
        n_clients: int,
        dataset_conf: dict
    ):
    """A routine to load and split data."""

    # load the dataset requested
    trainset, testset \
        = load_data(dataset_name=dataset_conf["DATASET_NAME"],
                    dataset_path=dataset_conf["DATASET_PATH"],
                    dataset_down=dataset_conf["DATASET_DOWN"],
                    random_seed=dataset_conf["RANDOM_SEED"]
                   )

    # split the dataset if requested
    if dataset_conf["SPLIT"]:
        train_splits, split_labels \
            = split_data(
                train_data = trainset,
                num_partitions = n_clients,
                split_method = dataset_conf["SPLIT_METHOD"],
                dirichlet_alpha = dataset_conf["DIRICHLET_ALPHA"], 
                random_seed = dataset_conf["RANDOM_SEED"], 
                min_partition_size = dataset_conf["MIN_PARTITION_SIZE"],
                classes_per_worker = dataset_conf["CLASSES_PER_WORKER"]
            )

        # Performing cleanups
        del trainset

        return (train_splits, split_labels), testset

    else:
        return (trainset, None), testset
