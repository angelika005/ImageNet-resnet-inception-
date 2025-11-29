from collections import Counter
from torchvision.datasets import ImageFolder
from data_loading import data_loading

dataset = ImageFolder(root='./ImageNet1000')
labels = [label for _, label in dataset]
class_counts = Counter(labels)

print(f"Всего классов: {len(class_counts)}")
print(f"Мин изображений: {min(class_counts.values())}")
print(f"Макс изображений: {max(class_counts.values())}")
print(f"Среднее: {sum(class_counts.values()) / len(class_counts):.1f}")

# Классы с очень малым количеством
small = [cls for cls, cnt in class_counts.items() if cnt < 50]
print(f"Классов с <50 изображений: {len(small)}")


train_loader, val_loader, _ = data_loading(batch_size=64, num_workers=8)
images, labels = next(iter(train_loader))

print(f"Batch images shape: {images.shape}")
print(f"Batch labels shape: {labels.shape}")
print(f"Labels min/max: {labels.min()}/{labels.max()}")
print(f"Unique labels: {len(labels.unique())}")
print(f"Image min/max: {images.min():.3f}/{images.max():.3f}")

# Проверка нормализации
print(f"Image mean: {images.mean():.3f} (ожидается ~0)")
print(f"Image std: {images.std():.3f} (ожидается ~1)")