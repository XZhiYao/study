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


class MyVgg16(nn.Module):
    def __init__(self):
        super(MyVgg16, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.features_conv = self.vgg.features[:30]
        self.max_pool = self.vgg.features[30]
        self.avgpool = self.vgg.avgpool
        self.classifier = self.vgg.classifier
        self.gradients = None

    def activation_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)
        h = x.register_hook(self.activation_hook)
        x = self.max_pool(x)
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)


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

    # LABELS_URL = "https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a.js"
    # response = requests.get(LABELS_URL)
    # labels = {int(key): value for key, value in response.json().item()}
    LABELS_DIR = "../DataFrame/imagenet1000_clsidx_to_labels.txt"
    labels = {}
    with open(LABELS_DIR, "r") as f:
        for data in f.readlines():
            data = data.strip('{')
            data = data.strip('}')
            data = data.strip('\n')
            (key, value) = data.strip().split(':')
            labels[int(key)] = value.replace("'", '')

    vgg16.eval()
    im_pre = vgg16(input_im)
    softmax = nn.Softmax(dim=1)
    im_pre_prob = softmax(im_pre)
    prob, prelab = torch.topk(im_pre_prob, 5)
    prob = prob.data.numpy().flatten()
    prelab = prelab.numpy().flatten()
    for ii, lab in enumerate(prelab):
        print("index: ", lab, "label: ", labels[lab], "||", prob[ii])

    print('--------------------------------------------------------------------------------------------')
    vggcam = MyVgg16()
    vggcam.eval()
    im_pre = vggcam(input_im)
    softmax = nn.Softmax(dim=1)
    im_pre_prob = softmax(im_pre)
    prob, prelab = torch.topk(im_pre_prob, 5)
    prob = prob.data.numpy().flatten()
    prelab = prelab.numpy().flatten()
    for ii, lab in enumerate(prelab):
        # print(ii)
        print("index: ", lab, "label: ", labels[lab], "||", prob[ii])

    # print('--------------------------------------------------------------------------------------------')
    im_pre[:, prelab[0]].backward()
    gradients = vggcam.get_activations_gradient()
    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = vggcam.get_activations(input_im).detach()
    for i in range(len(mean_gradients)):
        activations[:, i, :, :] *= mean_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.numpy()
    plt.matshow(heatmap)
    # plt.show()

    img = cv2.imread("../Dataset/image.jpg")
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    Grad_cam_img = heatmap * 0.4 + img
    Grad_cam_img = Grad_cam_img / Grad_cam_img.max()

    b, g, r = cv2.split(Grad_cam_img)
    Grad_cam_img = cv2.merge([r, g, b])
    plt.figure()
    plt.imshow(Grad_cam_img)
    plt.show()







