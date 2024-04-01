import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from data_setup import data_setup


def imshow(img, label, predicted, outputs):
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    true = "Same" if label == 1 else "Different"
    predicted = "Same" if predicted == 1 else "Different"
    axs[0].set_title(f"Truth: {true}, label: {label}")
    axs[1].set_title(f"Prediction: {predicted}, model output: {outputs:.2f}")
    axs[0].imshow(img[0].cpu().numpy(), cmap="gray")
    axs[1].imshow(img[1].cpu().numpy(), cmap="gray")
    axs[0].axis("off")
    axs[1].axis("off")
    plt.show()


def visualize_preds(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images1 = data["img0"].to(device)
            images2 = data["img1"].to(device)
            labels = data["label"].to(device)
            outputs = model(images1, images2).squeeze()

            predicted = np.where(outputs.cpu().detach().numpy() > 0.5, 1, 0)

            imshow(
                torch.cat((images1[0], images2[0]), 0),
                labels[0].cpu().numpy(),
                predicted[0],
                outputs[0].cpu().numpy(),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to model.pth file")
    parser.add_argument("config_path", help="Path to config.yaml file")
    args = parser.parse_args()
    data = data_setup(args, load_model=True)
    model = data["model"]
    test_loader = data["test_loader"]
    visualize_preds(model, test_loader)
