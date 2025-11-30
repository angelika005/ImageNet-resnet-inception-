config_resnet={
        "learning_rate": 0.025,
        "epochs": 2,
        "batch_size": 64,
        "architecture": "ResNet50",
        "optimizer": "SGD",
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "num_classes": 1000
    }

config_inceptionV3={
        "learning_rate": 0.025,
        "epochs": 1,
        "batch_size": 64,
        "architecture": "InceptionV3",
        "optimizer": "SGD",
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "num_classes": 1000,
        "aux_weight": 0.3,
}
