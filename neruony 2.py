import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# **1. Dane wejściowe: normalizacja i augmentacja**
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizacja
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# **2. W pełni połączony model z większą liczbą warstw i Dropout**
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FullyConnectedNN, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, 64),  # Więcej neuronów
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            nn.Linear(16, num_classes)  # Ostateczna warstwa
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Spłaszczenie obrazów
        return self.fc_layers(x)

model = FullyConnectedNN(input_size=3072, num_classes=10)

# **3. Ulepszony optymalizator z regularizacją L2 (Weight Decay)**
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# **4. Harmonogram zmiany tempa uczenia**
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# **5. Funkcja straty**
criterion = nn.CrossEntropyLoss()

# **6. Funkcja treningu**
def train_model(model, train_loader, optimizer, criterion, scheduler, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()  # Aktualizacja tempa uczenia
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# **7. Funkcja testowania**
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

# **8. Uruchomienie treningu i testu**
train_model(model, train_loader, optimizer, criterion, scheduler, num_epochs=10)
evaluate_model(model, test_loader)
