from custom_resnet_50 import ResNet, ResidualBlock
import torch.nn as nn
from torch import optim
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from data_loading import data_loading
import wandb
from config import config_resnet
import os

def load_model_for_test(filepath, device, num_classes=1000):
    model = ResNet(ResidualBlock, [3,4,6,3], num_classes=num_classes)
    model.load_state_dict(torch.load(filepath, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def top_k_accuracy(outputs, labels, k=5):
    _, topk_preds = outputs.topk(k, dim=1)
    correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
    return correct.sum().item() / labels.size(0)

def train_model(train_loader, val_loader):
    os.environ["WANDB_MODE"]="offline"
    save_dir = "saved_files_resnet"
    os.makedirs(save_dir, exist_ok=True)
    wandb.init(
    project="ImageNetClassification",
    config=config_resnet,
    name=f"resnet50_lr{config_resnet['learning_rate']}_bs{config_resnet['batch_size']}")

    num_epochs = config_resnet["epochs"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(ResidualBlock, [3,4,6,3], num_classes=config_resnet["num_classes"]).to(device)

    wandb.watch(model, log="all", log_freq=100)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config_resnet["learning_rate"], momentum=config_resnet["momentum"], weight_decay=config_resnet["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    print('Started training')

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

            if (i + 1) % 50 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch": epoch*len(train_loader)+i
                })

                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100*correct / total

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        top5_correct = 0
        val_total = 0

        print('Started validation')
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)

                _, topk_preds = outputs.topk(5, dim=1)
                correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
                top5_correct += correct.sum().item()
                val_total += labels.size(0)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        top5_accuracy = top5_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        wandb.log({
            "epoch": epoch+1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "top5_accuracy": top5_accuracy,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        print(f'  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        print(f'  Top-5 Accuracy: {top5_accuracy:.4f}')
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            #wandb.save(best_model_path)
            print(f"New best model saved, val_acc: {val_accuracy:.4f}")

        scheduler.step()

        if (epoch + 1) % 2 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_accuracy
            }, checkpoint_path)
            #wandb.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    #wandb.save(final_model_path)
    print("Final model saved")

    wandb.finish()

def test_model(test_loader, num_classes=1000):
    os.environ["WANDB_MODE"] = "offline"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model_path = os.path.join('saved_files_resnet', 'best_model.pth')
    model = load_model_for_test(best_model_path, device, num_classes=num_classes)
    model.eval()
    all_preds = []
    all_labels = []

    print("Testing model")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) 

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f'\nTest Results:')
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1 Score: {f1:.4f}')

    print("\n" + "="*50)
    print("Sample Predictions:")
    print("="*50)
    visualize_predictions(model, test_loader, device, num_samples=5)

    wandb.init(project="ImageNetClassification", job_type="evaluation")
    wandb.log({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1
    })
    wandb.finish()


def visualize_predictions(model, test_loader, device, num_samples=5):

    with open("names.yaml", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    
    model.eval()
    samples_shown = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            probs = torch.nn.functional.softmax(outputs, dim=1)
            top5_probs, top5_indices = probs.topk(5, dim=1)
            
            for i in range(min(inputs.size(0), num_samples - samples_shown)):
                true_label = labels[i].item()
                pred_label = top5_indices[i][0].item()
                
                print(f"\n--- Sample {samples_shown + 1} ---")
                print(f"True class: {class_names[true_label]} (ID: {true_label})")
                print(f"Predicted class: {class_names[pred_label]} (ID: {pred_label})")
                print(f"Correct: {'✓' if true_label == pred_label else '✗'}")
                print("\nTop-5 predictions:")
                for j in range(5):
                    idx = top5_indices[i][j].item()
                    prob = top5_probs[i][j].item()
                    print(f"  {j+1}. {class_names[idx]}: {prob*100:.2f}%")
                
                samples_shown += 1
                if samples_shown >= num_samples:
                    return

if __name__ == '__main__':  
    train_loader, val_loader, test_loader = data_loading(batch_size=config_resnet["batch_size"], num_workers=8)
    #train_model(train_loader, val_loader)
    test_model(test_loader, num_classes=config_resnet["num_classes"])