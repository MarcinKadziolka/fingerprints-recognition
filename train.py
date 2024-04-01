import wandb
import os
from config_manager import config
import shutil
from train_test_loops import train
from data_setup import data_setup
import argparse


def main(args):
    wandb.init(project="siamese-network", mode="offline")

    data = data_setup(args)

    net = data["model"]
    train_loader = data["train_loader"]
    test_loader = data["test_loader"]
    criterion = data["criterion"]
    optimizer = data["optimizer"]
    num_epochs = config.num_epochs

    print("Training Started")
    model = train(
        net, train_loader, test_loader, num_epochs, criterion, optimizer, wandb
    )
    model.save_model(os.path.join(wandb.run.dir, "model.pth"))
    shutil.copyfile(args.config_path, os.path.join(wandb.run.dir, args.config_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to config.yaml file")
    args = parser.parse_args()
    main(args)
