import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Definition
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
            nn.Linear(512 * 15 * 15, num_classes)  # Ensure input dimensions match
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

# Instantiate the model
num_classes = 5  # Replace with the number of classes in your dataset
model = ResNet9(3, num_classes).to(device)

# Load the model weights
model.load_state_dict(torch.load('C:/Users/06dev/PycharmProjects/newweather/resnet9_model.pth'))
model.eval()  # Set the model to evaluation mode
print("Model loaded successfully!")

# Transformations for the input image
transform = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.ToTensor()
])

# Load and preprocess the image
image_path = "C:/Users/06dev/PycharmProjects/newweather/img4.jpg"  # Replace with your image path
img = Image.open(image_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, dim=1)
    print("Predicted label:", predicted.item())
