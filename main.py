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
BATCH_SIZE = 20
MAX_PHYSICAL_BATCH_SIZE = 10
MAX_GRAD_NORM = 1.2
EPOCHS = 20
LR = 1e-3

# Privacy Budgets
use_differential_privacy = True
EPSILON = 10.0
DELTA = 1e-5

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"


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
    train_dl, test_dl, val_dl = get_dataloders()
    model = ChozenModel.getModel()
    if use_differential_privacy:
        print("Using Differential Privacy")
        if not isinstance(model, GradSampleModule):
            model = GradSampleModule(model)
        model = ModuleValidator.fix(model)

    optimizer = optim.SGD(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    if use_differential_privacy:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_dl = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_dl,
            epochs=EPOCHS,
            target_epsilon=EPSILON,
            target_delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
        )

    model = model.to(device)

    train_accuracies = []
    test_accuracies = []
    epsilons = []

    for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
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

            if use_differential_privacy:
                # Get epsilon
                epsilon = privacy_engine.get_epsilon(DELTA)
                epsilons.append(epsilon)
            print(f"Epoch: {epoch}")
            print(f"Train Accuracy: {train_accuracy}")
            print(f"Test Accuracy: {test_accuracy}")
            if use_differential_privacy:
                print(f"ε = {epsilon:.2f}, δ = {DELTA}")
    print(f"Train Accuracies = {train_accuracies}")
    print(f"Test Accuracies = {test_accuracies}")
    if use_differential_privacy:
        print(f"Epsilons = {epsilons}")

    torch.save(model.state_dict(), "./trained_models/model_weights.pth")


if __name__ == "__main__":
    main()
