"""
Test Module: Vgg16(torch)
Dataset: Kaggle-10-Monkey-Species
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import hiddenlayer as hl
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import torch.utils.data as Data
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder


class MyPreVggNet(nn.Module):
    def __init__(self):
        super(MyPreVggNet, self).__init__()
        self.vgg = vgg

        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=25088,
                out_features=512,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(
                in_features=512,
                out_features=256,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(
                in_features=256,
                out_features=10,
            ),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.vgg(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output


def view_dataset_batch(train_data_loader):
    for step, (b_x, b_y) in enumerate(train_data_loader):
        if step > 0:
            break

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array(([0.229, 0.224, 0.225]))
    plt.figure(figsize=(12, 6))
    for ii in np.arange(len(b_y)):
        plt.subplot(4, 8, ii+1)
        image = b_x[ii, :, :, :].numpy().transpose((1, 2, 0))
        image = std * image + mean
        image = np.clip(image, 0, 1)
        plt.imshow(image)
        plt.title(b_y[ii].data.numpy())
        plt.axis("off")
    plt.subplots_adjust(hspace=0.3)
    plt.show()


if __name__ == '__main__':
    vgg16 = models.vgg16(pretrained=True)
    vgg = vgg16.features
    for param in vgg.parameters():
        param.requires_grad_(False)

    myprevgg = MyPreVggNet()
    print(myprevgg)

    train_data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(), # p=0.5
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data_dir = "../Dataset/10-monkey-species/training"
    train_data = ImageFolder(train_data_dir, transform=train_data_transforms)
    train_data_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=32,
        shuffle=True,
        num_workers=2,
    )
    val_data_dir = "../Dataset/10-monkey-species/validation"
    val_data = ImageFolder(val_data_dir, transform=val_data_transforms)
    val_data_loader = Data.DataLoader(
        dataset=val_data,
        batch_size=32,
        shuffle=True,
        num_workers=2,
    )
    print("train sample: ", len(train_data.targets))
    print("val sample: ", len(val_data.targets))

    view_dataset_batch(train_data_loader)
