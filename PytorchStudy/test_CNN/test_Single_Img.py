import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import cv2
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from PIL import Image


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def view_image(imarray):
    plt.figure()
    plt.imshow(imarray)
    plt.show()


def view_superfical_featureMap(vgg16):
    vgg16.eval()
    vgg16.features[4].register_forward_hook(get_activation("maxpool1"))
    _ = vgg16(input_im)
    maxpool1 = activation["maxpool1"]
    print("get the feature map size: ", maxpool1.shape)

    plt.figure(figsize=(11, 6))
    for ii in range(maxpool1.shape[1]):
        plt.subplot(6, 11, ii + 1)
        plt.imshow(maxpool1.data.numpy()[0, ii, :, :], cmap="gray")
        plt.axis("off")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def view_deep_featureMap(vgg16):
    vgg16.eval()
    vgg16.features[21].register_forward_hook(get_activation("layer21_conv"))
    _ = vgg16(input_im)
    layer21_conv = activation["layer21_conv"]
    print("get the feature map size: ", layer21_conv.shape)

    plt.figure(figsize=(12, 6))
    for ii in range(72):
        plt.subplot(6, 12, ii + 1)
        plt.imshow(layer21_conv.data.numpy()[0, ii, :, :], cmap="gray")
        plt.axis("off")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


if __name__ == '__main__':
    vgg16 = models.vgg16(pretrained=True)
    im = Image.open("../Dataset/image.jpg")
    imarray = np.asarray(im) / 255.0
    # view_image(imarray)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_im = data_transforms(im).unsqueeze(0)
    print("input_im.shape: ", input_im.shape)

    # view_superfical_featureMap(vgg16)
    # view_deep_featureMap(vgg16)


