from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.transforms as transforms
import torch

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def data_preparation():
    imagenet_mean = (0.485, 0.456, 0.406) 
    imagenet_std  = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize(342),#resnet-256, inception-342
        transforms.RandomCrop(299),#resnet-224, inception-299
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(342),#resnet-256, inception-342
        transforms.CenterCrop(299),#resnet-224, inception-299
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean,
                         imagenet_std)
    ]) 
    return train_transform, val_test_transform

def data_loading(batch_size=64, num_workers=8):
    train_transform, val_test_transform = data_preparation()

    dataset = ImageFolder(root='./ImageNet1000', transform=None)

    total_len = len(dataset)
    train_len = int(0.7*total_len)
    val_len = int(0.2*total_len)
    test_len = total_len - train_len - val_len

    train_data, val_data, test_data = random_split(dataset, [train_len, val_len, test_len])

    train_set = TransformSubset(train_data, transform=train_transform)
    val_set = TransformSubset(val_data, transform=val_test_transform)
    test_set = TransformSubset(test_data, transform=val_test_transform)

    print(f'train_size = {len(train_set)}, val_size = {len(val_set)}, test_size = {len(test_set)}')

    train_loader =  DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader