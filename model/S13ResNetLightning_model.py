import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import torch.optim as optim
from torchmetrics import Accuracy
from torch.utils.data import random_split

from ERAV2_main.utils import CIFAR10ResNetUtil
from ERAV2_main.utils import CIFAR10AlbumenationDataSet


class BasicBlock(LightningModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(LightningModule):
    def __init__(self, block, num_blocks, lr_finder=True, num_classes=10):
        super(ResNet, self).__init__()
        self.cifar_train = None
        self.cifar_val = None
        self.cifar_test = None
        self.with_lr_finder = lr_finder

        self.util = CIFAR10ResNetUtil()
        self.criterion = nn.CrossEntropyLoss()
        self.trainTransform = self.util.get_train_transform_cifar10_resnet()
        self.testTransform = self.util.get_test_transform_cifar10_resnet()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        train_loss = self.criterion(output, target)
        predication = output.argmax(dim=1)
        self.accuracy(predication, target)

        self.log("train_loss", train_loss, prog_bar=True)
        self.log("train_acc", self.accuracy, prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        test_loss = self.criterion(output, target).item()
        predication = output.argmax(dim=1)
        self.accuracy(predication, target)

        self.log("val_loss", test_loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)

        return test_loss

    def test_step(self, test_batch, batch_idx):
        self.validation_step(test_batch, batch_idx)

    def configure_optimizers(self):
        if self.with_lr_finder:
            optimizer = optim.Adam(self.parameters(), lr=1e-10, weight_decay=1e-2)
            lr = self.util.find_lr_fastai(self, None, self.train_dataloader(), self.criterion, optimizer)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                steps_per_epoch=len(self.train_dataloader()),
                epochs=self.trainer.max_epochs,
                pct_start=.3,
                div_factor=100,
                three_phase=False,
                final_div_factor=100,
                anneal_strategy='linear'
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    'interval': 'step',  # or 'epoch'
                    'frequency': 1
                },
            }
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-2)
        return optimizer

    def prepare_data(self):
        CIFAR10AlbumenationDataSet('../data', train=True, download=True)
        CIFAR10AlbumenationDataSet('../data', train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        print('stage: ', stage)
        # if stage == "fit" or stage is None:
        full_cifar = CIFAR10AlbumenationDataSet('../data', train=True,
                                                                          transform=self.trainTransform)
        self.cifar_train, self.cifar_val = random_split(full_cifar, [0.7, 0.3])

        # Assign test dataset for use in dataloader(s)
        # if stage == "test" or stage is None:
        self.cifar_test = CIFAR10AlbumenationDataSet('../data', train=False, transform=self.testTransform)

    def train_dataloader(self):
        return self.util.get_data_loader_cifar10(self.cifar_train)

    def test_dataloader(self):
        return self.util.get_data_loader_cifar10(self.cifar_test)

    def val_dataloader(self):
        return self.util.get_data_loader_cifar10(self.cifar_val)

    def display_miss_classified_images(self):
        images = self.util.get_misclassified_images(self, test_loader=self.val_dataloader())
        self.util.plot_images(self.cifar_test, images, true_image=False)
        return images

    def display_gradcam(self, images, layer):
        self.util.show_grad_cam_heatmap(self, train_set=self.cifar_train, images=images, layer=layer)


def ResNet18(with_lr_finder=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], lr_finder=with_lr_finder)


def ResNet34(with_lr_finder=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], lr_finder=with_lr_finder)
