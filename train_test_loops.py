import torch
import numpy as np
from sklearn.metrics import accuracy_score
import tqdm
import matplotlib.pyplot as plt


def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    val_true = []
    val_pred = []
    val_loss = 0
    with torch.no_grad():
        for data in test_loader:
            images1 = data["img0"].to(device)
            images2 = data["img1"].to(device)
            labels = data["label"].to(device)
            outputs = model(images1, images2).squeeze()

            predicted = np.where(outputs.cpu().detach().numpy() > 0.5, 1, 0)

            loss = criterion(outputs, labels.float())
            val_loss += loss.item()

            val_true.extend(labels.cpu().numpy())
            val_pred.extend(predicted)

    val_acc = accuracy_score(val_true, val_pred)
    return val_acc, val_loss


def train(model, train_loader, test_loader, num_epochs, criterion, optimizer, wandb):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_progress = tqdm.tqdm(range(num_epochs))
    for epoch in epoch_progress:
        y_true = []
        y_pred = []
        train_loss = 0
        model.train()
        for _, data in enumerate(train_loader):
            images1 = data["img0"].to(device)
            images2 = data["img1"].to(device)
            labels = data["label"].to(device)

            outputs = model(images1, images2).squeeze()

            predicted = np.where(outputs.cpu().detach().numpy() > 0.9, 1, 0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted)

            loss = criterion(outputs, labels.float())
            train_loss += loss.item()

            # visualize(images1, images2, labels, outputs, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_acc = accuracy_score(y_true, y_pred)

        model.eval()
        val_true = []
        val_pred = []
        val_loss = 0
        with torch.no_grad():
            for data in test_loader:
                images1 = data["img0"].to(device)
                images2 = data["img1"].to(device)
                labels = data["label"].to(device)
                outputs = model(images1, images2).squeeze()

                predicted = np.where(outputs.cpu().detach().numpy() > 0.9, 1, 0)
                # visualize(images1, images2, labels, outputs, epoch)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()

                val_true.extend(labels.cpu().numpy())
                val_pred.extend(predicted)

        val_acc = accuracy_score(val_true, val_pred)

        epoch_progress.set_description(
            f"Epoch: [{epoch+1}/{num_epochs}], train accuracy: {train_acc:.4f}, train loss: {train_loss:.4f}, val_accuracy: {val_acc:.4f}, {val_loss:.4f}"
        )
        wandb.log(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
    return model


def visualize(images1, images2, labels, outputs, epoch):
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    for i in range(3):
        axs[i, 0].imshow(images1[i].squeeze().cpu().detach().numpy(), cmap="gray")
        axs[i, 1].imshow(images2[i].squeeze().cpu().detach().numpy(), cmap="gray")
        axs[i, 0].set_title(f"Same fingers: {labels[i]}")
        axs[i, 1].set_title(f"Predicted: {outputs[i]:.2f}")
        axs[i, 0].axis("off")
        axs[i, 1].axis("off")
    plt.show()
