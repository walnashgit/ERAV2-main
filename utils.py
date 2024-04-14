import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchsummary import summary
from Util.CIFAR10DataSet import CIFAR10AlbumenationDataSet
from Util.main import *
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import datasets, transforms

mean = (0.4914, 0.4822, 0.4465)
std_dev = (0.2471, 0.2435, 0.2616)


class CIFAR10ResNetUtil:

    def __init__(self, seed=1):
        self.cuda = False
        self.use_mps = False
        self.set_seed(seed)

    def set_seed(self, seed):
        self.cuda = torch.cuda.is_available()
        print("CUDA available: ", self.cuda)
        self.use_mps = torch.backends.mps.is_available()
        print("mps: ", self.use_mps)

        if self.cuda:
            torch.cuda.manual_seed(seed)
        elif self.use_mps:
            torch.mps.manual_seed(seed)
        else:
            torch.manual_seed(seed)

    def get_train_transform_cifar10_resnet(self):
        return A.Compose([
            A.PadIfNeeded(36, 36),
            A.RandomCrop(32, 32),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8,
                            fill_value=list(mean)),
            A.Normalize(mean, std_dev),
            ToTensorV2()
        ])

    def get_test_transform_cifar10_resnet(self):
        return A.Compose([
            A.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ToTensorV2()
        ])

    def get_test_transform_cifar10_resnet_2(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_train_set_cifar10(self, train_transforms=None):
        if train_transforms is None:
            train_transforms = self.get_train_transform_cifar10_resnet()
        return CIFAR10AlbumenationDataSet('../data', train=True, download=True, transform=train_transforms)

    def get_test_set_cifar10(self, test_transforms=None):
        if test_transforms is None:
            test_transforms = self.get_test_transform_cifar10_resnet()
        return CIFAR10AlbumenationDataSet('../data', train=False, download=True, transform=test_transforms)

    def get_data_loader_args(self):
        dataloader_args = dict(shuffle=True, batch_size=512)
        if self.cuda:
            dataloader_args = dict(shuffle=True, batch_size=512, num_workers=2, pin_memory=True)
        elif self.use_mps:
            dataloader_args = dict(shuffle=True, batch_size=512, pin_memory=True)
        return dataloader_args

    def get_data_loader_cifar10(self, data_set):
        dataloader_args = self.get_data_loader_args()
        return torch.utils.data.DataLoader(data_set, **dataloader_args)

    def get_available_device(self):
        device = torch.device("cpu")
        if self.cuda:
            device = torch.device("cuda")
        elif self.use_mps:
            device = torch.device("mps")
        return device

    def print_summary(self, model, device):
        if self.cuda:
            model.to(device)
        summary(model, input_size=(3, 32, 32))

    def find_lr_fastai(self, model, device, train_loader, criterion, optimizer):
        lr_finder = LRFinder(model, optimizer, criterion, device=device)
        lr_finder.range_test(train_loader, end_lr=100, num_iter=100, step_mode="exp")
        lr_finder.reset()
        return lr_finder.plot()[-1]

    def find_lr_leslie_smith(self, model, device, train_loader, test_loader, criterion, optimizer):
        lr_finder = LRFinder(model, optimizer, criterion, device=device)
        lr_finder.range_test(train_loader, val_loader=test_loader, end_lr=100, num_iter=100, step_mode="exp")
        lr_finder.reset()
        return lr_finder.plot(log_lr=True)[-1]

    @staticmethod
    def denormalize(img):
        for i in range(img.shape[0]):
            img[i] = (img[i] * std_dev[i]) + mean[i]
        return img

    @staticmethod
    def start_training_testing(epochs, collect_images, model, device, train_loader,
                               test_loader, optimizer, criterion, scheduler=None):
        model = model.to(device)
        for epoch in range(epochs):
            print("EPOCH:", epoch)
            print('current Learning Rate: ', optimizer.state_dict()["param_groups"][0]["lr"])
            train(model, device, train_loader, optimizer, epoch, criterion)
            if scheduler is not None:
                scheduler.step()
            test(model, device, test_loader, epoch, epochs, collect_images, criterion)

    def plot_miss_classified_images(self, train_set, num_images=10):
        self.plot_images(train_set, missclassified_images, num_images, False)

    def plot_images(self, train_set, images, num_images=10, true_image=True):
        fig = plt.figure(figsize=(5, 5))
        fig.subplots_adjust(wspace=0.8, hspace=0.8)
        for i in range(min(num_images, len(images))):
            image, true_label, predicted_label = images[i]
            ax = fig.add_subplot(5, 2, i + 1)
            image = CIFAR10ResNetUtil.denormalize(image)
            ax.imshow(np.transpose(image, (1, 2, 0)))
            if true_image:
                ax.set_title(f'{train_set.classes[true_label]}')
            else:
                ax.set_title(f'True: {train_set.classes[true_label]}, \n Predicted: {train_set.classes[predicted_label]}')
            ax.axis('off')
        if self.use_mps:
            plt.show()

    def visualize_images(self, train_set, num_images=10):
        dataiter = iter(train_set)
        images = []
        for i in range(num_images):
            imgs, labels = next(dataiter)
            images.append((imgs, labels, None))
        self.plot_images(train_set=train_set, images=images)

    def plot_training_and_test_loss_and_accuracy(self):
        # Plot the training and test loss and accuracy
        t = [t_items.item() for t_items in train_losses]
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0, 0].plot(t)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(test_acc)
        axs[1, 1].set_title("Test Accuracy")
        if self.use_mps:
            plt.show()

    def show_grad_cam_heatmap(self, model, train_set, images=missclassified_images, num_images=10):
        target_layers = [model.layer4[-1]]

        fig = plt.figure(figsize=(8, 8))
        fig.subplots_adjust(wspace=0.8, hspace=0.8)

        # Create GradCAM object
        cam = GradCAM(model=model, target_layers=target_layers)

        for i, (image, true_label, predicted_label) in enumerate(images[:num_images]):

            input_tensor = image.unsqueeze(0)  # Add batch dimension

            # Specify target classes
            target = [ClassifierOutputTarget(predicted_label.item())]

            # Generate CAMs
            grayscale_cam = cam(input_tensor=input_tensor, targets=target)

            # Visualize CAMs
            grayscale_cam = grayscale_cam[0, :]  # Select the CAM for the first (and only) image in the batch

            image = self.denormalize(image)
            image = np.transpose(image, (1, 2, 0))
            image = np.array(image)
            image = image.astype(np.float32)
            visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True, image_weight=0.8)

            ax = fig.add_subplot(5, 2, i + 1)
            ax.imshow(visualization)
            ax.set_title(f'True: {train_set.classes[true_label]}, \n Predicted: {train_set.classes[predicted_label]}')
            ax.axis('off')

        plt.show()

