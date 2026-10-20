import torch
from torch import nn
from torchvision import models


# ResNet
class TorchvisionResNet(nn.Module):
    def __init__(self, arch="resnet18", num_digits=5, num_classes=10):
        super().__init__()
        self.num_digits = num_digits
        self.num_classes = num_classes
        if arch == "resnet18":
            self.backbone = models.resnet18(weights=None)
        elif arch == "resnet34":
            self.backbone = models.resnet34(weights=None)
        else:
            raise ValueError(f"unknown arch {arch!r}, choose from ['resnet18', 'resnet34']")
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(
            in_features,
            num_classes * num_digits,
        )

    def forward(self, images):
        logits = self.backbone(images)
        return logits.reshape(images.shape[0], self.num_classes, self.num_digits)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()
        # 第一层
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        # 第二层
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        # 跳跃路径上的投影
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 跳跃连接
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 残差和shortcut形状不一致，用投影
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class CAPTCHAResNet(nn.Module):
    def __init__(
        self,
        block=BasicBlock,
        layers=(2, 2, 2, 2),
        num_digits=5,
        num_classes=10,
        image_size=(80, 260),
    ):
        super().__init__()
        self.num_digits = num_digits
        self.num_classes = num_classes
        self.in_channels = 64
        # stem: 卷积 + BN + ReLU + MaxPool
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # （B, 3, H, W) -> (B, 64, H/2, W/2)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # (B, 64, H/4, W/4)
        )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        h, w = image_size
        for _ in range(5):
            h = (h - 1) // 2 + 1
            w = (w - 1) // 2 + 1
        feature_dim = 512 * block.expansion * h * w  # 展平后的特征维度

        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, num_digits * num_classes),
        )

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, num_blocks, stride=1):
        out_channels = channels * block.expansion

        # 做投影
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        blocks = [block(self.in_channels, channels, stride, downsample)]
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            blocks.append(block(self.in_channels, channels))

        return nn.Sequential(*blocks)

    def forward(self, images):
        x = self.stem(images)  # [B, 64, H/4, W/4]
        x = self.layer1(x)  # [B, 64, H/4, W/4]
        x = self.layer2(x)  # [B, 128, H/8, W/8]
        x = self.layer3(x)  # [B, 256, H/16, W/16]
        x = self.layer4(x)  # [B, 512, H/32, W/32]
        logits = self.classifier(x)  # 展平 + 全连接 -> [B, num_digits * num_classes]
        # reshape 成 [B, num_classes, num_digits]
        return logits.reshape(images.shape[0], self.num_classes, self.num_digits)


MODELS = ("resnet18", "resnet34", "captcha_resnet18", "captcha_resnet34")


def create_model(name, image_size=(80, 260)):
    if name == "resnet18":
        return TorchvisionResNet("resnet18")
    if name == "resnet34":
        return TorchvisionResNet("resnet34")
    if name == "captcha_resnet18":
        return CAPTCHAResNet(layers=(2, 2, 2, 2), image_size=image_size)
    if name == "captcha_resnet34":
        return CAPTCHAResNet(layers=(3, 4, 6, 3), image_size=image_size)
    raise ValueError(f"unknown model {name!r}, choose from {MODELS}")


def compute_loss(logits, targets):
    B, num_classes, num_digits = logits.shape
    logits = logits.permute(0, 2, 1).reshape(B * num_digits, num_classes)
    targets = targets.reshape(B * num_digits)
    loss = nn.functional.cross_entropy(logits, targets)
    return loss
