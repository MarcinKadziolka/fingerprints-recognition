import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision as tv
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


class SiameseDataset(Dataset):
    def __init__(
        self,
        csv_file=None,
        img_dir=None,
        transform=None,
        transform_candidate=None,
        pad=None,
    ):
        # used to prepare the labels and images path
        self.data = pd.read_csv(csv_file)
        self.pad_val = pad
        self.img_dir = img_dir
        self.transform = transform
        self.transform_candidate = transform_candidate

    def pad(self, image, pad):
        desired_width = 512
        desired_height = 512

        width, height = image.size
        padding_width = max(desired_width - width, 0)
        padding_height = max(desired_height - height, 0)
        left_padding = padding_width // 2
        top_padding = padding_height // 2
        right_padding = padding_width - left_padding
        bottom_padding = padding_height - top_padding

        padded_image = ImageOps.expand(
            image,
            border=(left_padding, top_padding, right_padding, bottom_padding),
            fill="white",
        )
        return padded_image

    def __getitem__(self, index):
        # getting the image path
        image_0_path = os.path.join(
            self.img_dir, "SOCOFing/Real", self.data.iat[index, 0]
        )

        image_1_path = os.path.join(
            self.img_dir, "SOCOFing/Altered/Altered-Easy", self.data.iat[index, 1]
        )
        if not os.path.exists(image_1_path):
            image_1_path = os.path.join(
                self.img_dir, "SOCOFing/Real", self.data.iat[index, 1]
            )

        label = self.data.iat[index, 2]
        # Loading the image
        img0 = Image.open(image_0_path)
        img1 = Image.open(image_1_path)

        if self.pad_val is not None:
            img1 = self.pad(img1, self.pad_val)

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)

        if self.transform_candidate is not None:
            img1 = self.transform_candidate(img1)

        data = {
            "img0": img0,
            "img1": img1,
            "label": label,
            "img0_path": image_0_path,
            "img1_path": image_1_path,
        }
        return data

    def __len__(self):
        return len(self.data)


def plot_data(data):
    x0 = data["img0"]
    x1 = data["img1"]
    label = data["label"]
    x0_path = data["img0_path"]
    x1_path = data["img1_path"]
    print(x0_path, x1_path)
    print(x0.shape, x1.shape)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(x0.squeeze())
    axs[1].imshow(x1.squeeze())
    plt.title(label)
    plt.show()


def print_data(data):
    img0 = data["img0"]
    img1 = data["img1"]
    img0_path = data["img0_path"]
    img1_path = data["img1_path"]
    print(img0.shape, img1.shape)
    print(img0_path, img1_path)
    assert img0.shape == img1.shape


def main():
    # For testing purposes
    transform = tv.transforms.Compose(
        [tv.transforms.ToTensor(), tv.transforms.Resize((100, 100))]
    )
    dataset = SiameseDataset(train=True, dir="dataset", transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for data in dataloader:
        print_data(data)
        plot_data(data)


if __name__ == "__main__":
    main()
