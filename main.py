import lightly
import pytorch_lightning as pl
import torch
import torchvision

from loss import BarlowTwinsLoss
from resnet50 import resnet50
from utils import BenchmarkModule

num_workers = 8
max_epochs = 800
knn_k = 200
knn_t = 0.1
classes = 10
batch_size = 256
seed = 1

pl.seed_everything(seed)

# use a GPU if available
gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
device = 'cuda' if gpus else 'cpu'

# Use SimCLR augmentations, additionally, disable blur
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=32,
    gaussian_blur=0.,
)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

dataset_train_ssl = lightly.data.LightlyDataset.from_torch_dataset(
    torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        download=True))
dataset_train_kNN = lightly.data.LightlyDataset.from_torch_dataset(torchvision.datasets.CIFAR10(
    root='data',
    train=True,
    transform=test_transforms,
    download=True))
dataset_test = lightly.data.LightlyDataset.from_torch_dataset(torchvision.datasets.CIFAR10(
    root='data',
    train=False,
    transform=test_transforms,
    download=True))

dataloader_train_ssl = torch.utils.data.DataLoader(
    dataset_train_ssl,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers
)
dataloader_train_kNN = torch.utils.data.DataLoader(
    dataset_train_kNN,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)


class BartonTwins(BenchmarkModule):
    def __init__(self, dataloader_kNN, gpus, classes, knn_k, knn_t):
        super().__init__(dataloader_kNN, gpus, classes, knn_k, knn_t)
        # create a ResNet backbone and remove the classification head

        self.backbone = resnet50(pretrained=False)
        # create a simsiam model based on ResNet
        # note that bartontwins has the same architecture
        self.resnet_simsiam = lightly.models.SimSiam(self.backbone, num_ftrs=2048, num_mlp_layers=3
                                                     # , out_dim=8192
                                                     )
        self.criterion = BarlowTwinsLoss(device=device)

    def forward(self, x):
        self.resnet_simsiam(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        (z_a, _), (z_b, _) = self.resnet_simsiam(x0, x1)
        loss = self.criterion(z_a, z_b)
        self.log('train_loss_ssl', loss)
        return loss

    # learning rate warm-up
    def optimizer_steps(self,
                        epoch=None,
                        batch_idx=None,
                        optimizer=None,
                        optimizer_idx=None,
                        optimizer_closure=None,
                        on_tpu=None,
                        using_native_amp=None,
                        using_lbfgs=None):
        # 120 steps ~ 1 epoch
        if self.trainer.global_step < 1000:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 1000.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * 1e-3

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simsiam.parameters(), lr=1e-3,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


model = BartonTwins(dataloader_train_kNN, gpus=gpus, classes=classes, knn_k=knn_k, knn_t=knn_t)

print(model)
trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus,
                     progress_bar_refresh_rate=100)
trainer.fit(
    model,
    train_dataloader=dataloader_train_ssl,
    val_dataloaders=dataloader_test
)

print(f'Highest test accuracy: {model.max_accuracy:.4f}')
