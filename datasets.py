import torchvision
from lightly import data as lightly_data
from torch.utils.data import DataLoader


def dataset_loader(name='cifar10', batch_size=256, num_workers=0):
    # Use SimCLR augmentations, additionally, disable blur
    collate_fn = lightly_data.SimCLRCollateFunction(
        input_size=32,
        gaussian_blur=0.,
    )

    # No additional augmentations for the test set
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=lightly_data.collate.imagenet_normalize['mean'],
            std=lightly_data.collate.imagenet_normalize['std'],
        )
    ])
    if name == 'cifar10':
        dataset_cls = torchvision.datasets.CIFAR10
    elif name == 'cifar100':
        dataset_cls = torchvision.datasets.CIFAR100
    else:
        raise Exception({'message': f'Unsupported Dataset {name}'})
    dataset_train_ssl = lightly_data.LightlyDataset.from_torch_dataset(
        torchvision.datasets.CIFAR10(
            root='data',
            train=True,
            download=True))
    dataset_train_kNN = lightly_data.LightlyDataset.from_torch_dataset(dataset_cls(
        root='data',
        train=True,
        transform=test_transforms,
        download=True))
    dataset_test = lightly_data.LightlyDataset.from_torch_dataset(dataset_cls(
        root='data',
        train=False,
        transform=test_transforms,
        download=True))

    dataloader_train_ssl = DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers
    )
    dataloader_train_kNN = DataLoader(
        dataset_train_kNN,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    return dataloader_train_ssl, dataloader_train_kNN, dataloader_test
