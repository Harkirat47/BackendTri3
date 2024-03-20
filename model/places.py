import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from collections import Counter
from PIL import Image

class ImageClassifier:
    def __init__(self, dataset_path, img_width=128, img_height=128, batch_size=32):
        self.dataset_path = dataset_path
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size

        self.data_transforms = transforms.Compose([
            transforms.Resize((self.img_width, self.img_height)),
            transforms.CenterCrop(self.img_width),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.label_vocab = None
        self.image_dataset = None
        self.data_loader = None
        self.model = None

    def _preprocess_data(self):
        images = []
        labels = []
        max_images_per_place = 12
        for location_folder in os.listdir(self.dataset_path):
            location_folder_path = os.path.join(self.dataset_path, location_folder)
            if os.path.isdir(location_folder_path):
                count = 0
                for image_file in os.listdir(location_folder_path):
                    if count >= max_images_per_place:
                        break
                    image_path = os.path.join(location_folder_path, image_file)
                    image = self.data_transforms(Image.open(image_path))
                    images.append(image)
                    labels.append(location_folder)
                    count += 1

        label_counter = Counter(labels)
        self.label_vocab = {label: i for i, label in enumerate(label_counter)}
        labels = [self.label_vocab[label] for label in labels]

        images = torch.stack(images)
        labels = torch.tensor(labels)

        self.image_dataset = torch.utils.data.TensorDataset(images, labels)
        self.data_loader = DataLoader(self.image_dataset, batch_size=self.batch_size, shuffle=True)

    def _initialize_model(self):
        num_classes = len(self.label_vocab)
        self.model = models.resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def train_model(self, num_epochs=20):
        self._preprocess_data()
        self._initialize_model()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in self.data_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(self.image_dataset)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    def predict_image_class(self, image_path):
        image = Image.open(image_path)
        image = self.data_transforms(image).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
            return predicted.item()


def initPlaces():
    global ImageClassifier
    places_classfier = ImageClassifier()
    places_classfier._initialize_model()
    places_classfier.train_model(num_epochs=20)




# classifier = ImageClassifier(dataset_path='/Users/shubhay/Documents/GitHub/BackendTri3/places')
# classifier.train_model()
# predicted_class = classifier.predict_image_class("/content/drive/MyDrive/ML/Data/places/Golden_Gate_Bridge/00b4e41b02.jpg")
# print("Predicted class:", predicted_class)
