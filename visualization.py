import matplotlib.pyplot as plt
import numpy as np
from data_loading import train_set, train_data
from dict_names import dict_names

def imshow(img_tensor):
    img = img_tensor.numpy().transpose((1, 2, 0))

    imagenet_mean = ([0.485, 0.456, 0.406])
    imagenet_std  = ([0.229, 0.224, 0.225])

    img = imagenet_std*img + imagenet_mean
    img = np.clip(img, 0, 1)

    plt.imshow(img)
    plt.axis('off')
    plt.show()

for i in range (5):
    img, label = train_set[i]
    print(f'Label: {label}, Class name: {dict_names[label]}')
    imshow(img)