from config_manager import config
from siamese_network import SiameseNetwork
from siamese_dataset import SiameseDataset
import torch.nn as nn
import torchvision as tv
import torch
import os
from torch.utils.data import DataLoader


def data_setup(args, load_model=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = args.config_path
    config.load(config_path)
    size = config.data.size
    model = SiameseNetwork(input_dims=(config.channels, size, size)).to(device)

    if load_model:
        model.load_model(args.model_path)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    size = config.data.size

    transform = tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.Resize((size, size)),
        ]
    )

    transform_candidate = tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.RandomRotation(degrees=config.data.random_rotation.degrees),
            tv.transforms.RandomAffine(
                degrees=config.data.random_affine.degrees,
                shear=config.data.random_affine.shear,
                scale=config.data.random_affine.scale,
            ),
            tv.transforms.RandomPerspective(
                distortion_scale=config.data.random_perspective.distortion_scale,
                p=config.data.random_perspective.p,
            ),
            tv.transforms.CenterCrop(size=(size, size)),
            tv.transforms.RandomApply(
                [
                    tv.transforms.GaussianBlur(
                        kernel_size=config.data.gaussian_blur.kernel_size
                    ),
                    tv.transforms.RandomResizedCrop(
                        size=(size, size),
                        scale=config.data.random_resized_crop.scale,
                    ),
                ],
                p=config.data.random_resized_crop.p,
            ),
        ]
    )

    csv_dir = "csv_dir"
    img_dir = "preprocessed"
    pad = config.data.pad
    train_dataset = SiameseDataset(
        csv_file=os.path.join(csv_dir, "train.csv"),
        img_dir=img_dir,
        transform=transform,
        transform_candidate=transform_candidate,
        pad=pad,
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=config.batch_size, num_workers=4
    )

    test_dataset = SiameseDataset(
        csv_file=os.path.join(csv_dir, "test.csv"),
        img_dir=img_dir,
        transform=transform,
        transform_candidate=transform_candidate,
        pad=pad,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
    )

    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "train_loader": train_loader,
        "test_loader": test_loader,
    }
