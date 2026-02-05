import torch
import torch.nn as nn
from torchvision import models


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet18Mel(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(planes=64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(planes=128, blocks=2, stride=2)
        self.layer3 = self._make_layer(planes=256, blocks=2, stride=2)
        self.layer4 = self._make_layer(planes=512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)

        if pretrained:
            self._load_pretrained_resnet18_weights()

    def _make_layer(self, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(BasicBlock(in_planes=self.inplanes, planes=planes,stride=stride, downsample=downsample))
        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(BasicBlock(in_planes=self.inplanes,planes=planes,stride=1,downsample=None))

        return nn.Sequential(*layers)

    def _load_pretrained_resnet18_weights(self):
        try:
            tv_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except AttributeError: 
            tv_model = models.resnet18(pretrained=True)

        tv_state = tv_model.state_dict()
        for k in ["conv1.weight", "fc.weight", "fc.bias"]:
            if k in tv_state:
                tv_state.pop(k)
        self.load_state_dict(tv_state, strict=False)

        with torch.no_grad():
            conv1_rgb = tv_model.conv1.weight        
            conv1_gray = conv1_rgb.mean(dim=1, keepdim=True)  
            self.conv1.weight.copy_(conv1_gray)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)         
        x = torch.flatten(x, 1)    
        x = self.fc(x)               
        return x


def resnet18(pretrained):
    return ResNet18Mel(pretrained=pretrained)