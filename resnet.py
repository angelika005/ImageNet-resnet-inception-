import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from data_loading import data_loading
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os

def top_k_accuracy(outputs, labels, k=5):
    _, topk_preds = outputs.topk(k, dim=1)
    correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
    return correct.sum().item() / labels.size(0)

def load_model_for_test(filepath, device, num_classes=10):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_checkpoint(model, optimizer, scheduler, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        return start_epoch
    else:
        return 0

def train_model(train_loader, val_loader):
    model_resnet = models.resnet50(weights=None)
    num_classes = 10
    model_resnet.fc = torch.nn.Linear(model_resnet.fc.in_features, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_resnet = model_resnet.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_resnet.parameters(), lr=0.001, momentum=0.9, weight_decay = 0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    start_epoch = load_checkpoint(model_resnet, optimizer, scheduler, 'checkpoint_epoch_20.pth')

    num_epochs = 30
    print('started learning')

    for epoch in range(start_epoch, start_epoch+num_epochs):
        model_resnet.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model_resnet(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{start_epoch+num_epochs}], '
                      f'Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{start_epoch+num_epochs}] Training Loss: {avg_train_loss:.4f}')

        model_resnet.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        top5_correct = 0
        total = 0

        print('started validation')
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model_resnet(inputs)
                _, topk_preds = outputs.topk(5, dim=1)
                correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
                top5_correct += correct.sum().item()
                total += labels.size(0)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        top5_accuracy = top5_correct / total
        print(f'Validation Loss: {val_loss/len(val_loader):.3f}, Accuracy: {accuracy:.4f},\
                Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Top-5 accuracy: {top5_accuracy:.4f}')
        
        scheduler.step()

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model_resnet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, f'checkpoint_epoch_{epoch+1}.pth')
    print("Checkpoint saved")
    torch.save(model_resnet.state_dict(), 'final_model.pth')
    print("Final model saved to final_model.pth")


def test_model(test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_resnet = load_model_for_test('final_model.pth', device)
    model_resnet.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_resnet(inputs)
            _, preds = torch.max(outputs, 1) 

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1 Score: {f1:.4f}')

if __name__ == '__main__':  
    train_loader, val_loader, test_loader = data_loading()
    #train_model(train_loader, val_loader)
    test_model(test_loader)
