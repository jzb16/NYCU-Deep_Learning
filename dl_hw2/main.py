import torch
import matplotlib.pyplot as plt
from dataloader import BufferflyMothLoader
from ResNet50 import resnet50
from VGG19 import vggnet
import torch.optim as optim
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BufferflyMothLoader
train_dataset = BufferflyMothLoader(root='./dataset', mode='train')
test_dataset = BufferflyMothLoader(root='./dataset', mode='test')
valid_dataset = BufferflyMothLoader(root='./dataset', mode='valid')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Setting Model, optimizers
models = {'ResNet50': resnet50().to(device), 'VGGNet': vggnet().to(device)}
#models = { 'VGGNet': vggnet().to(device)}
criterion = nn.CrossEntropyLoss()
optimizers = {name: optim.Adam(model.parameters(), lr=0.001) for name, model in models.items()}
num_epochs = 100

# accuracy history & loss history
accuracy_history = {name: {'train_acc': [], 'test_acc': [],'valid_acc': []} for name in models.keys()}
loss_history = {name: {'train_loss': [], 'test_loss': [],'valid_loss': []} for name in models.keys()}

# Train
def train(model, optimizer, num_epochs, train_loader, model_name):
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_corrects = 0
        train_loss = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0) 
            train_loss += loss.item()
            train_corrects += torch.sum(predicted == labels.data)
        
        epoch_acc = (train_corrects.double() / len(train_loader.dataset)) * 100
        epoch_loss = (train_loss / len(train_loader.dataset)) * 100
        accuracy_history[model_name]['train_acc'].append(epoch_acc.cpu().numpy())
        loss_history[model_name]['train_loss'].append(epoch_loss)
        print(f'Epoch {epoch}/{num_epochs}, {model_name} Train Acc: {epoch_acc} / Train Loss: {epoch_loss}')
        
        # Save model weight
        torch.save(model.state_dict(), f'{model_name}_weights.pth')

# Test
def test(model, test_loader, num_epochs, model_name):
    for epoch in range(num_epochs):
        model.eval()
        test_correct = 0
        test_loss = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                test_correct += (predicted == target).sum().item()

        test_acc = (test_correct / total) * 100
        epoch_loss = (test_loss / total) * 100
        accuracy_history[model_name]['test_acc'].append(test_acc)
        loss_history[model_name]['test_loss'].append(epoch_loss)

    print(f'Test Acc: {test_acc}')

def valid(model, valid_loader, num_epochs, model_name, models_weights):
    model.load_state_dict(torch.load(models_weights))
    for epoch in range(1):
        model.eval()
        valid_correct = 0
        valid_loss = 0
        total = 0

        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                valid_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                valid_correct += (predicted == target).sum().item()

        valid_acc = (valid_correct / total) * 100
        epoch_loss = (valid_loss / total) * 100
        accuracy_history[model_name]['valid_acc'].append(valid_acc)
        loss_history[model_name]['valid_loss'].append(epoch_loss)

    print(f'Valid Acc: {valid_acc}')

# Evaluate
def evaluate(models, optimizers, num_epochs, train_loader, test_loader, models_weights):
    for name, model in models.items():
        model_weight = models_weights[name]
        print("----------------------------------------------------------")
        optimizer = optimizers[name]
        #print(f"Training {name}")
        #train(model, optimizer, num_epochs, train_loader, name)
        #print()
        #print("----------------------------------------------------------")
        #print(f"Testing {name}")
        #test(model, test_loader, num_epochs, name)
        #print()
        #print("----------------------------------------------------------")
        print(f"Validating {name}")
        valid(model, valid_loader, num_epochs, name, model_weight)
        print()

models_weights = {'ResNet50': './weight_60/ResNet50_weights_88.pth', 'VGGNet': './weight_100/VGGNet_weights.pth'}

# Train and test
evaluate(models, optimizers, num_epochs, train_loader, test_loader, models_weights)

# Accuracy plot
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
for name in accuracy_history:
    plt.plot(accuracy_history[name]['train_acc'], label=f'{name} Train Acc')
    #plt.plot(accuracy_history[name]['test_acc'], label=f'{name} Test Acc')
    plt.plot(accuracy_history[name]['valid_acc'], label=f'{name} Valid Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# loss plot
plt.subplot(2, 1, 2)
for name in loss_history:
    plt.plot(loss_history[name]['train_loss'], label=f'{name} Train Loss')
    #plt.plot(accuracy_history[name]['test_loss'], label=f'{name} Test Loss')
    plt.plot(loss_history[name]['valid_loss'], label=f'{name} Valid Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("Results1.png")
plt.show()
