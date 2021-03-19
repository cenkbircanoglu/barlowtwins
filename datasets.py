import torchvision
from lightly.data import LightlyDataset, collate, SimCLRCollateFunction
from torch.utils.data import DataLoader


def dataset_loader(name='cifar10', batch_size=256, num_workers=0, data_root='./data'):
    # Use SimCLR augmentations, additionally, disable blur
    collate_fn = SimCLRCollateFunction(
        input_size=32,
        gaussian_blur=0.,
    )

    # No additional augmentations for the test set
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=collate.imagenet_normalize['mean'],
            std=collate.imagenet_normalize['std'],
        )
    ])
    if name == 'cifar10':
        dataset_cls = torchvision.datasets.CIFAR10
    elif name == 'cifar100':
        dataset_cls = torchvision.datasets.CIFAR100
    # TODO add support for LSUN dataset
    # elif name == 'lsun':
    #    dataset_cls = torchvision.datasets.LSUN
    else:
        raise Exception({'message': f'Unsupported Dataset {name}'})
    dataset_train_ssl = LightlyDataset.from_torch_dataset(
        dataset_cls(
            root=data_root,
            train=True,
            download=True)
    )
    dataset_train_kNN = LightlyDataset.from_torch_dataset(
        dataset_cls(
            root=data_root,
            train=True,
            transform=test_transforms,
            download=True)
    )
    dataset_test = LightlyDataset.from_torch_dataset(
        dataset_cls(
            root=data_root,
            train=False,
            transform=test_transforms,
            download=True)
    )

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
