import torch.nn as nn
import torch
import torch.nn.functional as F

class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem = StemBlock()

        self.inception_a1 = InceptionA(192, 32)
        self.inception_a2 = InceptionA(256, 64)
        self.inception_a3 = InceptionA(288, 64)

        self.reduction_a = ReductionA(288)

        self.inception_b1 = InceptionB(768, 128)
        self.inception_b2 = InceptionB(768, 160)
        self.inception_b3 = InceptionB(768, 160)
        self.inception_b4 = InceptionB(768, 192)

        self.aux_classifier = AuxiliaryClassifier(768, num_classes)

        self.reduction_b = ReductionB(768)

        self.inception_c1 = InceptionC(1280)
        self.inception_c2 = InceptionC(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, 2048)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a1(x)
        x = self.inception_a2(x)
        x = self.inception_a3(x)
        x = self.reduction_a(x)
        x = self.inception_b1(x)
        x = self.inception_b2(x)
        x = self.inception_b3(x)
        x = self.inception_b4(x)
        aux = self.aux_classifier(x)
        x = self.reduction_b(x)
        x = self.inception_c1(x)
        x = self.inception_c2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x, aux

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return F.relu(x, inplace=True)
    

class StemBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNReLU(3, 32, 3, stride=2, padding=0)
        self.conv2 = ConvBNReLU(32, 32, 3, stride=1, padding=0)
        self.conv3 = ConvBNReLU(32, 64, 3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=0)
        self.conv4 = ConvBNReLU(64, 80, 1, stride=1, padding=0)
        self.conv5 = ConvBNReLU(80, 192, 3, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool2(x)

        return x


class InceptionA(nn.Module):
    def __init__(self, in_channels, nbr_kernels):
        super().__init__()

        self.branch1_conv1 = ConvBNReLU(in_channels, 64, 1, padding=0)
        self.branch1_conv2 = ConvBNReLU(64, 96, 3, padding=1)
        self.branch1_conv3 = ConvBNReLU(96, 96, 3, padding=1)

        self.branch2_conv1 = ConvBNReLU(in_channels, 48, 1, padding=0)
        self.branch2_conv2 = ConvBNReLU(48, 64, 3, padding=1)

        self.branch3_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.branch3_conv = ConvBNReLU(in_channels, nbr_kernels, 1, padding = 0)

        self.branch4_conv = ConvBNReLU(in_channels, 64, 1, padding=0)

    def forward(self, x):
        b1 = self.branch1_conv1(x)
        b1 = self.branch1_conv2(b1)
        b1 = self.branch1_conv3(b1)

        b2 = self.branch2_conv1(x)
        b2 = self.branch2_conv2(b2)

        b3 = self.branch3_pool(x)
        b3 = self.branch3_conv(b3)

        b4 = self.branch4_conv(x)

        return torch.cat([b1, b2, b3, b4], dim=1)


class InceptionB(nn.Module):
    def __init__(self, in_channels, nbr_kernels):
        super().__init__()

        self.branch1_conv1 = ConvBNReLU(in_channels, nbr_kernels, 1, padding=0)
        self.branch1_conv2 = ConvBNReLU(nbr_kernels, nbr_kernels, (7,1), padding=(3,0))
        self.branch1_conv3 = ConvBNReLU(nbr_kernels, nbr_kernels, (1,7), padding=(0, 3))
        self.branch1_conv4 = ConvBNReLU(nbr_kernels, nbr_kernels, (7,1), padding=(3,0))
        self.branch1_conv5 = ConvBNReLU(nbr_kernels, 192, (1,7), padding=(0,3))
        
        self.branch2_conv1 = ConvBNReLU(in_channels, nbr_kernels, 1, padding=0)
        self.branch2_conv2 = ConvBNReLU(nbr_kernels, nbr_kernels, (1,7), padding=(0,3))
        self.branch2_conv3 = ConvBNReLU(nbr_kernels, 192, (7,1), padding=(3,0))
        
        self.branch3_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.branch3_conv = ConvBNReLU(in_channels, 192, 1, padding=0)
        
        self.branch4_conv = ConvBNReLU(in_channels, 192, 1, padding=0)
        
    def forward(self, x):
        b1 = self.branch1_conv1(x)
        b1 = self.branch1_conv2(b1)
        b1 = self.branch1_conv3(b1)
        b1 = self.branch1_conv4(b1)
        b1 = self.branch1_conv5(b1)
        
        b2 = self.branch2_conv1(x)
        b2 = self.branch2_conv2(b2)
        b2 = self.branch2_conv3(b2)
        
        b3 = self.branch3_pool(x)
        b3 = self.branch3_conv(b3)
        
        b4 = self.branch4_conv(x)
        
        return torch.cat([b1, b2, b3, b4], dim=1)
    

class InceptionC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1_conv1 = ConvBNReLU(in_channels, 448, 1, padding=0)
        self.branch1_conv2 = ConvBNReLU(448, 384, 3, padding=1)
        self.branch1_conv3a = ConvBNReLU(384, 384, (1,3), padding=(0,1))
        self.branch1_conv3b = ConvBNReLU(384, 384, (3,1), padding=(1,0))
        
        self.branch2_conv1 = ConvBNReLU(in_channels, 384, 1, padding=0)
        self.branch2_conv2a = ConvBNReLU(384, 384, (1,3), padding=(0,1))
        self.branch2_conv2b = ConvBNReLU(384, 384, (3,1), padding=(1,0))
        
        self.branch3_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.branch3_conv = ConvBNReLU(in_channels, 192, 1, padding=0)
        
        self.branch4_conv = ConvBNReLU(in_channels, 320, 1, padding=0)
        
    def forward(self, x):
        b1 = self.branch1_conv1(x)
        b1 = self.branch1_conv2(b1)
        b1a = self.branch1_conv3a(b1)
        b1b = self.branch1_conv3b(b1)
        b1 = torch.cat([b1a, b1b], dim=1)
        
        b2 = self.branch2_conv1(x)
        b2a = self.branch2_conv2a(b2)
        b2b = self.branch2_conv2b(b2)
        b2 = torch.cat([b2a, b2b], dim=1)
        
        b3 = self.branch3_pool(x)
        b3 = self.branch3_conv(b3)
        
        b4 = self.branch4_conv(x)
        
        return torch.cat([b1, b2, b3, b4], dim=1)


class ReductionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1_conv1 = ConvBNReLU(in_channels, 64, 1, padding=0)
        self.branch1_conv2 = ConvBNReLU(64, 96, 3, padding=1)
        self.branch1_conv3 = ConvBNReLU(96, 96, 3, stride=2, padding=0)
        
        self.branch2_conv = ConvBNReLU(in_channels, 384, 3, stride=2, padding=0)

        self.branch3_pool = nn.MaxPool2d(3, stride=2, padding=1)
        
    def forward(self, x):
        b1 = self.branch1_conv1(x)
        b1 = self.branch1_conv2(b1)
        b1 = self.branch1_conv3(b1)
        
        b2 = self.branch2_conv(x)
        
        b3 = self.branch3_pool(x)
        
        return torch.cat([b1, b2, b3], dim=1)

class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1_conv1 = ConvBNReLU(in_channels, 192, 1, padding=0)
        self.branch1_conv2 = ConvBNReLU(192, 192, (1,7), padding=(0,3))
        self.branch1_conv3 = ConvBNReLU(192, 192, (7,1), padding=(3,0))
        self.branch1_conv4 = ConvBNReLU(192, 192, 3, stride=2, padding=0)
        
        self.branch2_conv1 = ConvBNReLU(in_channels, 192, 1, padding=0)
        self.branch2_conv2 = ConvBNReLU(192, 320, 3, stride=2, padding=0)
        
        self.branch3_pool = nn.MaxPool2d(3, stride=2, padding=0)
        
    def forward(self, x):
        b1 = self.branch1_conv1(x)
        b1 = self.branch1_conv2(b1)
        b1 = self.branch1_conv3(b1)
        b1 = self.branch1_conv4(b1)
        
        b2 = self.branch2_conv1(x)
        b2 = self.branch2_conv2(b2)
        
        b3 = self.branch3_pool(x)
        
        return torch.cat([b1, b2, b3], dim=1)
    

class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=1000):
        super().__init__()
        self.avgpool = nn.AvgPool2d(5, stride=3, padding=0)
        self.conv = ConvBNReLU(in_channels, 128, 1, padding=0)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4,4))
        self.fc1 = nn.Linear(2048, 768)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(768, num_classes)
        
    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x