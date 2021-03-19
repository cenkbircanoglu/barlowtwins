import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from lightly import models
from omegaconf import DictConfig

from datasets import dataset_loader
from loss import BarlowTwinsLoss
from models.resnet50 import resnet50
from utils import BenchmarkModule

# use a GPU if available
gpus = 1 if torch.cuda.is_available() else 0
device = 'cuda' if gpus else 'cpu'

torch.multiprocessing.set_sharing_strategy('file_system')


class BartonTwins(BenchmarkModule):
    def __init__(self, dataloader_kNN, gpus, classes, knn_k, knn_t, num_ftrs=512, backbone=None, max_epochs=1):
        super().__init__(dataloader_kNN, gpus, classes, knn_k, knn_t)
        self.backbone = backbone
        # create a simsiam model based on ResNet
        # note that bartontwins has the same architecture
        self.resnet_simsiam = models.SimSiam(self.backbone, num_ftrs=num_ftrs, num_mlp_layers=3)
        self.criterion = BarlowTwinsLoss(device=device)
        self.max_epochs = max_epochs

    def forward(self, x):
        self.resnet_simsiam(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        (z_a, _), (z_b, _) = self.resnet_simsiam(x0, x1)
        # our simsiam model returns both (features + projection head)
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]


def get_backbone(name):
    if name in ['resnet-9', 'resnet-18', 'resnet-50', 'resnet-101']:
        resnet = models.ResNetGenerator(name)
        return nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
    if name == 'dilated-resnet-50':
        return resnet50(pretrained=False)


@hydra.main(config_path='./conf', config_name="resnet_18")
def run_app(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    dataloader_train_ssl, dataloader_train_kNN, dataloader_test = dataset_loader(name=cfg.dataset_name,
                                                                                 batch_size=cfg.batch_size,
                                                                                 num_workers=cfg.num_workers,
                                                                                 data_root=cfg.data_root)
    backbone = get_backbone(cfg.backbone_name)
    model = BartonTwins(dataloader_train_kNN, gpus=gpus, classes=cfg.classes, knn_k=cfg.knn_k, knn_t=cfg.knn_t,
                        num_ftrs=cfg.num_ftrs, backbone=backbone, max_epochs=cfg.max_epochs)
    print(model)
    trainer = pl.Trainer(max_epochs=cfg.max_epochs, gpus=gpus,
                         progress_bar_refresh_rate=100)
    trainer.fit(
        model,
        train_dataloader=dataloader_train_ssl,
        val_dataloaders=dataloader_test,
    )

    print(f'Highest test accuracy: {model.max_accuracy:.4f}')


if __name__ == '__main__':
    run_app()
