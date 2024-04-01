import argparse
from train_test_loops import test
from data_setup import data_setup


def main(args):
    data = data_setup(args, load_model=True)
    model = data["model"]
    test_loader = data["test_loader"]
    criterion = data["criterion"]

    print("Test started")
    test_acc, test_loss = test(model, test_loader, criterion)
    print(f"Test accuracy: {test_acc}")
    print(f"Test loss: {test_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to model.pth file")
    parser.add_argument("config_path", help="Path to config.yaml file")
    args = parser.parse_args()
    main(args)
