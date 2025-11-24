import torchvision.transforms as transforms

#basic
def data_preparation():
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize(342),#256
        transforms.RandomCrop(299),#224
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(342),#256
        transforms.CenterCrop(299),#224
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean,
                         imagenet_std)
    ]) 
    return train_transform, val_test_transform