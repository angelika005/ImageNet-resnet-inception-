from dict_names import dict_names as names
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
from data_preparation import data_preparation
from PIL import Image

def data_loading():
    train_transform, val_test_transform = data_preparation()

    dataset = ImageFolder(root='./ImageNet10', transform=None)

    train_dataset = ImageFolder(root='./ImageNet10', transform=train_transform)
    val_dataset = ImageFolder(root='./ImageNet10', transform=val_test_transform)
    test_dataset = ImageFolder(root='./ImageNet10', transform=val_test_transform)

    total_len = len(dataset)
    train_len = int(0.7*total_len)
    val_len = int(0.2*total_len)
    test_len = total_len - train_len - val_len

    train_data, val_data, test_data = random_split(dataset, [train_len, val_len, test_len])

    train_set = Subset(train_dataset, train_data.indices)
    val_set = Subset(val_dataset, val_data.indices)
    test_set = Subset(test_dataset, test_data.indices)

    print(f'train_size = {len(train_set)}, val_size = {len(val_set)}, test_size = {len(test_set)}')

    train_loader =  DataLoader(train_set, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader