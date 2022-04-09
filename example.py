import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from models import ViT

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.CenterCrop(224)
    ])
    train_set = torchvision.datasets.Flowers102("./data", split="train", download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_set = torchvision.datasets.Flowers102("./data", split="test", download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=True)
    val_set = torchvision.datasets.Flowers102("./data", split="val", download=True, transform=transform)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("Model will be trained on device:", device)

    model = ViT(input_size=(224, 224), patch_size=(16, 16), num_classes=102).to(device)
    optimiser = optim.Adam(model.parameters(), weight_decay=0.1)
    loss_function = nn.CrossEntropyLoss()
    num_epoch = 10

    train_loss_list = []
    test_loss_list = []
    val_loss_list = []

    for epoch in range(10):
        running_loss = 0
        for data in train_loader:
            model.train()
            inputs, labels = data
            labels = labels - 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimiser.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
        print("epoch %d/%d:(tr)loss=%.4f" % (epoch, num_epoch, running_loss))
        val_loss_list.append(running_loss)
        running_loss = 0
        total = 0
        correct = 0
        for data in val_loader:
            with torch.no_grad():
                model.eval()
                inputs, labels = data
                labels = labels - 1
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                loss = loss_function(outputs, labels)
                running_loss += loss.item()

                pred = torch.argmax(F.softmax(outputs), dim=1)
                total += len(labels)
                correct += sum(pred == labels)

        print("epoch %d/%d:(va)loss=%.4f" % (epoch, num_epoch, running_loss))
        print("epoch %d/%d:(va)acc=%.4f%%" % (epoch, num_epoch, (100.0 * correct) / total))
        val_loss_list.append(running_loss)

    torch.save(model.state_dict(), "test/vit")
    print("Saved.")
