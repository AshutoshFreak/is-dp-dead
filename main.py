import argparse
import models.FullResNet.FullResNet as ChozenModel
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
from PathMNIST import PathMNIST
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from tqdm import tqdm
import warnings

warnings.simplefilter("ignore")

# Hyperparameters
MAX_GRAD_NORM = 1.2

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Command line parameters.")
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs (default: 20)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--use_differential_privacy",
        action="store_true",
        default=False,
        help="Use differential privacy (default: False)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=10.0,
        help="Epsilon value (default: 10.0). Required if use_differential_privacy is True",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Delta value (default: 1e-5). Required if use_differential_privacy is True",
    )
    parser.add_argument(
        "--save_model_path", type=str, default="", help="Save directory for model"
    )
    parser.add_argument(
        "--pretrained_path", type=str, default="", help="Load pretrained model"
    )

    args = parser.parse_args()

    # Check for required arguments if differential privacy is used
    if args.use_differential_privacy:
        if args.epsilon < 0:
            raise ValueError("Epsilon must be a positive float.")
        if args.delta < 0:
            raise ValueError("Delta must be a positive float.")

    return args


def accuracy(preds, labels):
    return (preds == labels).mean()


def get_dataloders():
    light_transform = tt.Compose(
        [tt.ToTensor(), tt.Lambda(lambda x: (x - x.mean()) / (x.std()))]
    )

    train_ds = PathMNIST(split="train", transforms=[light_transform])
    val_ds = PathMNIST(split="val", transforms=[light_transform])
    test_ds = PathMNIST(split="test", transforms=[light_transform])
    train_dl = DataLoader(train_ds, 20, shuffle=True, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, 20, True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, 20, True, num_workers=4, pin_memory=True)

    return train_dl, test_dl, val_dl


def train(model, criterion, optimizer, train_loader, epoch, device):
    model.train()
    for i, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def main():
    args = parse_args()
    train_dl, test_dl, val_dl = get_dataloders()
    model = ChozenModel.getModel(pretrained_path=args.pretrained_path)
    if args.use_differential_privacy:
        print("Using Differential Privacy.")
        print(f"Target Epsilon = {args.epsilon}")
        print(f"Target Delta = {args.delta}")
        if not isinstance(model, GradSampleModule):
            model = GradSampleModule(model)
        model = ModuleValidator.fix(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if args.use_differential_privacy:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_dl = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_dl,
            epochs=args.epochs,
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            max_grad_norm=MAX_GRAD_NORM,
        )

    model = model.to(device)

    train_accuracies = []
    test_accuracies = []
    epsilons = []

    for epoch in tqdm(range(args.epochs), desc="Epoch", unit="epoch"):
        train(model, criterion, optimizer, train_dl, epoch + 1, device)

        # Calculate test and train accuracy for each epoch
        correct = 0
        total = 0

        with torch.no_grad():
            # Calculate train accuracy
            for images, labels in train_dl:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            train_accuracy = correct / total * 100
            train_accuracies.append(train_accuracy)

            # Calculate test accuracy
            for images, labels in test_dl:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_accuracy = correct / total * 100
            test_accuracies.append(test_accuracy)

            if args.use_differential_privacy:
                # Get epsilon
                epsilon = privacy_engine.get_epsilon(args.delta)
                epsilons.append(epsilon)
            print(f"Epoch: {epoch}")
            print(f"Train Accuracy: {train_accuracy}")
            print(f"Test Accuracy: {test_accuracy}")
            if args.use_differential_privacy:
                print(f"ε = {epsilon:.2f}, δ = {args.delta}")
    print(f"Train Accuracies = {train_accuracies}")
    print(f"Test Accuracies = {test_accuracies}")
    if args.use_differential_privacy:
        print(f"Epsilons = {epsilons}")

    if args.save_model_path != "":
        torch.save(
            model.state_dict(),
            f"{args.save_model_path}/FullResNet_e{args.epsilon}.pth",
        )


if __name__ == "__main__":
    main()
