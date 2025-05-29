import os
import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from numba import jit


@jit(nopython=True)
def normalize_image(image):
    return image/255.0

class SignLanguageDataset(Dataset):
    def __init__(self, images, labels, transform = None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
def load_dataset(dataset_dir):
    images = []
    labels = []
    class_labels = os.listdir(dataset_dir)

    for label in class_labels:
        class_dir = os.path.join(dataset_dir, label)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            img = cv2.resize(img, (64, 64))   #why 64?
            img_array = normalize_image(img)
            images.append(img_array)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels, class_labels

dataset_path = 'C:\\Users\\Viviktha\\Downloads\\dataset\\dataset' #eyyyyyyyy it worked
images, labels, class_labels = load_dataset(dataset_path)

label_encoder = LabelEncoder() #initiates the labelencoder class from sklearn preprocessing module
labels_encoded = label_encoder.fit_transform(labels) #learns the unique classes/categories present in the labels array and then transforms categorical labels to numerical values

X_train, X_temp, y_train, y_temp = train_test_split(images, labels_encoded, test_size = 0.3, random_state = 42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 42)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #mean and standard deviation set to 0.5 for RGB channels
])

train_dataset = SignLanguageDataset(X_train, y_train, transform)
val_dataset = SignLanguageDataset(X_val, y_val, transform)
test_dataset = SignLanguageDataset(X_test, y_test, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle = False) #not shuffled bcoz it may lead to inconsistent results if shuffled
test_loader = DataLoader(test_dataset, batch_size=32, shuffle = False) #not shuffled bcoz it may lead to inconsistent results if shuffled

class CNNModel(nn.Module): # nn.Module is base class from whcih CNNModel inherits functionalities
    def __init__(self, num_classes):
        super(CNNModel, self).__init__() # calls the constructor of parent class nn.module to initialise 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels =32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding =1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128*8*8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using device: {device}')

num_classes = len(class_labels)
model = CNNModel(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

# Tracking metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        images = images.float()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation metrics calculation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.float()
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Plotting loss and accuracy curves
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()

# Test evaluation
model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        images = images.float()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')
cm = confusion_matrix(all_labels, all_predictions)

print(f"Test Precision: {100 * precision:.2f}")
print(f"Test Recall: {100 * recall:.2f}")
print(f"Test F1 Score: {100 * f1:.2f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, target_names=class_labels))

display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
display.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Save the model
model_path = 'C:\\Users\\Viviktha\\OneDrive\\Desktop\\Projects\\ISL Project\\model_weights.pth'
torch.save(model.state_dict(), model_path)
