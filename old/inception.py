import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from data_loading import data_loading
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os
import glob

def cleanup_old_checkpoints(keep_last_n=5):
    checkpoints = sorted(glob.glob('checkpoint_epoch_inception_*.pth'))
    if len(checkpoints) > keep_last_n:
        for old_checkpoint in checkpoints[:-keep_last_n]:
            os.remove(old_checkpoint)
            print(f"Deleted old checkpoint: {old_checkpoint}")

def top_k_accuracy(outputs, labels, k=5):
    _, topk_preds = outputs.topk(k, dim=1)
    correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
    return correct.sum().item() / labels.size(0)

def load_model_for_test(filepath, device, num_classes=10):
    model = models.inception_v3(weights=None, aux_logits=True, init_weights=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    model.load_state_dict(torch.load(filepath, map_location=device, weights_only=False))
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
    model_inception = models.inception_v3(weights=None, aux_logits=True)
    num_classes = 10
    model_inception.fc = torch.nn.Linear(model_inception.fc.in_features, num_classes)
    model_inception.AuxLogits.fc = torch.nn.Linear(model_inception.AuxLogits.fc.in_features, num_classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_inception = model_inception.to(device)
    print(f"Using device: {device}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model_inception.parameters(), lr=0.045, momentum=0.9, weight_decay = 0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    start_epoch = load_checkpoint(model_inception, optimizer, scheduler, 'checkpoint_epoch_inception_30.pth')

    num_epochs = 30
    print('started learning')

    for epoch in range(start_epoch, start_epoch+num_epochs):
        model_inception.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model_inception(inputs)

            if model_inception.training and model_inception.aux_logits:
                loss1 = criterion(outputs.logits, labels)
                loss2 = criterion(outputs.aux_logits, labels)
                loss = loss1 + 0.4 * loss2
            else:
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

        model_inception.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        top5_correct = 0
        total = 0

        print('Started validation')
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model_inception(inputs)
                
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
        
        print(f'Epoch [{epoch+1}/{start_epoch+num_epochs}] '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Accuracy: {accuracy:.4f}, '
              f'Precision: {precision:.4f}, '
              f'Recall: {recall:.4f}, '
              f'F1: {f1:.4f}, '
              f'Top-5: {top5_accuracy:.4f}')
        
        checkpoint_path = f'checkpoint_epoch_inception_{epoch+1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model_inception.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss / len(val_loader),
            'val_accuracy': accuracy
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        cleanup_old_checkpoints(keep_last_n=5)
        
        scheduler.step()

    torch.save(model_inception.state_dict(), 'final_model_inception.pth')
    print("Final model saved to final_model_inception.pth")


def test_model(test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_inception = load_model_for_test('final_model_inception.pth', device)
    model_inception.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_inception(inputs)
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
