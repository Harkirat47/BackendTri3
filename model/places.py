import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import cv2
from torch.utils.data import DataLoader
from collections import Counter
from PIL import Image


dataset_path = '/Users/shubhay/Documents/GitHub/BackendTri3/places'

img_width, img_height = 128, 128
batch_size = 32

data_transforms = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.CenterCrop(img_width),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

images = []
labels = []

max_images_per_place = 12

for location_folder in os.listdir(dataset_path):
    location_folder_path = os.path.join(dataset_path, location_folder)
    if os.path.isdir(location_folder_path):
        # Traverse through each image in the location folder
        count = 0
        for image_file in os.listdir(location_folder_path):
            if count >= max_images_per_place:
                break
            image_path = os.path.join(location_folder_path, image_file)
            # Load and transform the image
            image = data_transforms(Image.open(image_path))
            images.append(image)
            labels.append(location_folder)  # Assuming folder name is the label
            count += 1

label_counter = Counter(labels)
label_vocab = {label: i for i, label in enumerate(label_counter)}
labels = [label_vocab[label] for label in labels]

images = torch.stack(images)
labels = torch.tensor(labels)

image_dataset = torch.utils.data.TensorDataset(images, labels)


data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)

num_classes = len(label_counter)

model = models.resnet50(pretrained=True)
num_features = model.fc.in_features

model.fc = nn.Linear(num_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_model(model, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(image_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

train_model(model, criterion, optimizer)

def predict_image_class(image_path):
    image = Image.open(image_path)
    image = data_transforms(image).unsqueeze(0)  # Apply transformations directly
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()


predicted_class = predict_image_class("/content/drive/MyDrive/ML/Data/places/Golden_Gate_Bridge/00b4e41b02.jpg")
print("Predicted class:", predicted_class)
