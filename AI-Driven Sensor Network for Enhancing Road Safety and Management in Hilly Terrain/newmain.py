import os
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Paths
data_dir = "C:/Users/06dev/Downloads/archive(15)/dataset2"
train_dir = os.path.join(data_dir, 'Training')
test_dir = os.path.join(data_dir, 'Testing')

train_transformations = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.ToTensor()
])
test_transformations = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.ToTensor()
])

training_dataset = ImageFolder(train_dir, transform=train_transformations)
testing_dataset = ImageFolder(test_dir, transform=test_transformations)

print("Class Mapping:")
for idx, class_name in enumerate(training_dataset.classes):
    print(f"{idx}: {class_name}")
print("\n")

val_size = 250
train_size = len(training_dataset) - val_size
train_ds, val_ds = random_split(training_dataset, [train_size, val_size])

batch_size = 16
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size * 2, shuffle=False, num_workers=0)

def show_example(img, label):
    print('Label:', training_dataset.classes[label])
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = self._conv_block(in_channels, 64)
        self.conv2 = self._conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(self._conv_block(128, 128), self._conv_block(128, 128))
        self.conv3 = self._conv_block(128, 256, pool=True)
        self.conv4 = self._conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(self._conv_block(512, 512), self._conv_block(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(512 * 15 * 15, num_classes)
        )

    def _conv_block(self, in_channels, out_channels, pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        if pool: layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

model_resnet = ResNet9(3, len(training_dataset.classes)).to(device)

epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_resnet.parameters(), lr=0.001)

for epoch in range(epochs):
    model_resnet.train()
    train_loss = 0
    correct_preds = 0
    total_preds = 0

    for batch in train_dl:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        outputs = model_resnet(images)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        correct_preds += torch.sum(preds == labels).item()
        total_preds += labels.size(0)

        for i in range(len(labels)):
            print(f"Predicted Class: {training_dataset.classes[preds[i]]}, Actual Label: {training_dataset.classes[labels[i]]}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_accuracy = 100 * correct_preds / total_preds
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_dl)}, Accuracy: {train_accuracy}%")

    model_resnet.eval()
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for batch in val_dl:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model_resnet(images)

            _, preds = torch.max(outputs, 1)

            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)

            for i in range(len(labels)):
                print(f"Validation Predicted Class: {training_dataset.classes[preds[i]]}, Actual Label: {training_dataset.classes[labels[i]]}")

    # Calculate and print accuracy for validation
    val_accuracy = 100 * correct_preds / total_preds
    print(f"Validation Accuracy: {val_accuracy}%")
    print("************************************")

torch.save(model_resnet.state_dict(), "resnet9_model_1_1.pth")
print("Model saved to resnet9_model.pth")
