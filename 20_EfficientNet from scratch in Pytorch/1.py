# ==================================================
#       Implementing EfficientNet from scratch      
# ==================================================

import torch
import torch.nn as nn
from math import ceil

base_model = [
    # expand_ration, channels, repeat, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3]
]

phi_values = {
    "b0": (0, 224, 0.2), 
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5)
}

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))


class SqueezeExitation(nn.Module):
    def __init__(self, in_channels, reduce_dim):
        super(SqueezeExitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduce_dim, 1),
            nn.SELU(),
            nn.Conv2d(reduce_dim, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ration, reduction=4):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prop = 0.8
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = int(in_channels * expand_ration)  # Corrected here
        self.expand = in_channels != hidden_dim

        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels, hidden_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.expand_conv = None

        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim
            ),
            SqueezeExitation(hidden_dim, reduce_dim=hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)

def create_features(self, width_factor, depth_factor, last_channels):
    channels = int(32 * width_factor)
    features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
    in_channels = channels

    for expand_ratio, channels, repeats, stride, kernel_size in base_model:
        out_channels = 4 * ceil(int(channels * width_factor) / 4)
        layers_repeats = ceil(repeats * depth_factor)

        for layer in range(layers_repeats):
            features.append(
                InvertedResidualBlock(in_channels, out_channels, expand_ration=expand_ratio, stride=stride if layer == 0 else 1, kernel_size=kernel_size, padding=kernel_size//2)
            )
            in_channels = out_channels  # Update in_channels here after each layer

    features.append(CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0))

    return nn.Sequential(*features)

        

class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, drop_out_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(drop_out_rate),
            nn.Linear(last_channels, num_classes)
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate
    
    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(in_channels, out_channels, expand_ration=expand_ratio, stride=stride if layer == 0 else 1, kernel_size=kernel_size, padding=kernel_size//2)
                )

            in_channels = out_channels

        features.append(CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0))

        return nn.Sequential(*features)
    
    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))
    
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    version = "b0"
    phi, res, drop_rate = phi_values[version]
    num_examples, num_classes = 4, 10
    x = torch.randn((num_examples, 3, res, res)).to(device=device)
    model = EfficientNet(version=version, num_classes=num_classes).to(device=device)
    print(model(x).shape)

if __name__ == "__main__":
    main()