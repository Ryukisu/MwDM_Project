import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. Przygotowanie danych
transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # Przycinanie i zmiana rozmiaru
    transforms.RandomHorizontalFlip(),  # Losowe odbicie w poziomie
    transforms.RandomRotation(15),  # Losowy obrót o ±15 stopni
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Zmiany koloru
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),  # Normalizacja
    transforms.Lambda(lambda x: x.view(-1))  # Spłaszczenie obrazów
])

batch_size = 128

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 2. Definicja modelu
# class FullyConnectedNN(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(FullyConnectedNN, self).__init__()
#         self.fc_layers = nn.Sequential(
#             nn.Linear(input_size, 1024),
#             nn.ReLU(),
#             # nn.Dropout(0.3),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             # nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             # nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             # nn.Dropout(0.3),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             # nn.Dropout(0.3),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, x):
#         return self.fc_layers(x)


# input_size = 32 * 32 * 3  # CIFAR-10 ma obrazy o wymiarach 32x32x3
# num_classes = 10
# model = FullyConnectedNN(input_size, num_classes)

class CNNWithFullyConnected(nn.Module):
    def __init__(self, num_classes):
        super(CNNWithFullyConnected, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 16x16x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8x64
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),  # 8x8x128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 4x4x128
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(4 * 4 * 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)  # Zmieniamy kształt danych na 3x32x32
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Spłaszczamy dane
        x = self.fc_layers(x)
        return x


num_classes = 10
model = CNNWithFullyConnected(num_classes)


# 3. Definicja funkcji straty i optymalizatora
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.001)


# 4. Trening modelu
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # Reset gradientów
            optimizer.zero_grad()

            # Przewidywania
            outputs = model(images)

            # Obliczanie straty
            loss = criterion(outputs, labels)
            loss.backward()

            # Aktualizacja wag
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")
        evaluate_model(model, test_loader)


# 5. Testowanie modelu
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")


# Uruchomienie treningu i testu
train_model(model, train_loader, criterion, optimizer, num_epochs=15)
evaluate_model(model, test_loader)
