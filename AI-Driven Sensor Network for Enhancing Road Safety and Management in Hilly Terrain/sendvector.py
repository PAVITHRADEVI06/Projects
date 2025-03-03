import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import json

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
model.load_state_dict(torch.load('C:/Users/06dev/PycharmProjects/newweather/resnet9_model_1_1.pth'))
model.eval()  # Set the model to evaluation mode
print("Model loaded successfully!")

# Class-to-Label Mapping
classes = ['Cloudy', 'Foggy', 'Rainy', 'Shine', 'Sunrise']  # Replace with your dataset's class names
print("Class-to-Label Mapping:")
for idx, class_name in enumerate(classes):
    print(f"Label {idx}: {class_name}")

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
    predicted_label = predicted.item()
    predicted_class_name = classes[predicted_label]
    print(f"Predicted Label: {predicted_label}")
    print(f"Predicted Class: {predicted_class_name}")

# GeoJSON creation function
def create_weather_geojson(output_path, predicted_weather, location_address, coordinates):
    """
    Create a GeoJSON file for weather predictions.

    Args:
        output_path (str): Path to save the GeoJSON file.
        predicted_weather (str): Weather prediction (e.g., "Rainy").
        location_address (str): Address of the location (e.g., "London, UK").
        coordinates (tuple): Coordinates (longitude, latitude) as a tuple (e.g., (-0.1276, 51.5072)).
    """
    # Construct the GeoJSON structure
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": coordinates,  # Longitude, Latitude
                },
                "properties": {
                    "Weather": predicted_weather,
                    "Address": location_address,
                },
            }
        ],
    }

    # Save GeoJSON to a file
    with open(output_path, "w") as geojson_file:
        json.dump(geojson_data, geojson_file, indent=4)

    print(f"GeoJSON saved successfully at: {output_path}")

# Path to save the GeoJSON file
output_geojson_path = "C:/Users/06dev/PycharmProjects/newweather/weather_prediction.geojson"

# Location details
location_address = ""  # Replace with the actual address
coordinates = ()  # Longitude, Latitude

# Create and save the GeoJSON with the predicted weather
create_weather_geojson(output_geojson_path, predicted_class_name, location_address, coordinates)

##make it keep on running in a loop and also when you get camera module make sure to send it directly to this code
