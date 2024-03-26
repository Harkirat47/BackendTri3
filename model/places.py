import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from collections import Counter
from PIL import Image
from io import BytesIO

class ImageClassifier:
    def __init__(self, dataset_path, img_width=128, img_height=128, batch_size=32):
        ## initialize variables, standardize image height and width
        self.dataset_path = dataset_path
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size

        ## converts all image to same size, centers, data processing, etc.
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
        ## gets data, first iterates through the path of the dataset provided.
        for location_folder in os.listdir(self.dataset_path):
            ## generates full path from root by combining base with folder
            location_folder_path = os.path.join(self.dataset_path, location_folder)
            if os.path.isdir(location_folder_path):
                count = 0
                ## selects every file from the image, up to the specific max
                for image_file in os.listdir(location_folder_path):
                    if count >= max_images_per_place:
                        break
                    image_path = os.path.join(location_folder_path, image_file)
                    ## opens each image path as a PIL image
                    image = self.data_transforms(Image.open(image_path))
                    ## adds image to image list
                    images.append(image)
                    ## adds to label as one of categories that shall be classifird
                    labels.append(location_folder)
                    count += 1

        ## section below initializes data and prepares for model
        label_counter = Counter(labels)
        self.label_vocab = {label: i for i, label in enumerate(label_counter)}
        labels = [self.label_vocab[label] for label in labels]

        images = torch.stack(images)
        labels = torch.tensor(labels)

        self.image_dataset = torch.utils.data.TensorDataset(images, labels)
        self.data_loader = DataLoader(self.image_dataset, batch_size=self.batch_size, shuffle=True)

    def _initialize_model(self):
        ## sets basic information of the model, i.e. number of images we want to classify, sets for Linear
        num_classes = len(self.label_vocab)
        self.model = models.resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def train_model(self, num_epochs=1):
        ## process data, set initial variables
        self._preprocess_data()
        self._initialize_model()

        ## specific optimizer and criteria using torch
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        ## train model
        self.model.train()
        ## determines loss for each epoch, loss should be decreasing with each epoch
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in self.data_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            ## determine loss for entire epoch, divides loss by length of dataset
            epoch_loss = running_loss / len(self.image_dataset)
            ## display epoch and loss
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    def predict_image_class(self, image):
        image = self.data_transforms(image).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
            # Convert predicted to a standard Python datatype
            predicted_item = predicted.item()
            return predicted_item



def initPlaces():
    global ImageClassifier
    places_classfier = ImageClassifier()
    places_classfier._initialize_model()
    places_classfier.train_model(num_epochs=1)




# classifier = ImageClassifier(dataset_path='/Users/shubhay/Documents/GitHub/BackendTri3/places')
# classifier.train_model()
# predicted_class = classifier.predict_image_class("/content/drive/MyDrive/ML/Data/places/Golden_Gate_Bridge/00b4e41b02.jpg")
# print("Predicted class:", predicted_class)
