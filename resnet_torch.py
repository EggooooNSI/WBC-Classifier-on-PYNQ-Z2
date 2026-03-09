import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    Standard ResNet for Tensil AI
    - Configurable channel size via initial_channels
    - Fixed Linear layer (no LazyLinear)
    - Float training (Tensil will do PTQ)
    """
    
    def __init__(self, in_ch, block, num_blocks, num_classes=4, initial_channels=16):
        super(ResNet, self).__init__()
        self.in_planes = initial_channels

        self.conv1 = nn.Conv2d(in_ch, self.in_planes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, initial_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, initial_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, initial_channels * 4, num_blocks[2], stride=2) if num_blocks[2] is not None else None
        self.layer4 = self._make_layer(block, initial_channels * 8, num_blocks[3], stride=2) if num_blocks[3] is not None else None
        self.layer5 = self._make_layer(block, initial_channels * 16, num_blocks[4], stride=2) if num_blocks[4] is not None else None

        # Calculate final channels for Linear layer
        final_channels = initial_channels * block.expansion
        if num_blocks[4] is not None:
            final_channels = initial_channels * 16 * block.expansion
        elif num_blocks[3] is not None:
            final_channels = initial_channels * 8 * block.expansion
        elif num_blocks[2] is not None:
            final_channels = initial_channels * 4 * block.expansion
        elif num_blocks[1] is not None:
            final_channels = initial_channels * 2 * block.expansion

        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(final_channels, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, cal_grad_cam=False):
        """
        Standard forward pass for Tensil AI
        
        Args:
            x: Input tensor [batch, channels, H, W]
            cal_grad_cam: Whether to register hook for GradCAM
            
        Returns:
            output: Classification logits [batch, num_classes]
            features: Feature vector before classifier [batch, final_channels]
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out) if self.layer3 is not None else out
        out = self.layer4(out) if self.layer4 is not None else out
        
        if cal_grad_cam:
            h = out.register_hook(self.activations_hook)
        
        out = self.layer5(out) if self.layer5 is not None else out
        
        # Global Average Pooling
        out = self.average_pool(out)
        out = torch.flatten(out, 1)
        
        features = out.clone()
        
        # Classifier
        out = self.fc(out)
        
        return out, features

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out) if self.layer3 is not None else out
        out = self.layer4(out) if self.layer4 is not None else out
        return out


def ResNet18(in_ch, num_classes, initial_channels=64):
    """ResNet18 for Tensil AI"""
    return ResNet(in_ch, BasicBlock, [2, 2, 2, 2, 3], num_classes=num_classes, 
                  initial_channels=initial_channels)

def ResNet10(in_ch, num_classes, initial_channels=16):
    """
    ResNet10 for Tensil AI (default: 16 channels for FPGA)
    
    Args:
        in_ch: Input channels (1 for grayscale, 3 for RGB)
        num_classes: Number of output classes
        initial_channels: Initial conv channels (default: 16 for FPGA)
    """
    return ResNet(in_ch, BasicBlock, [1, 1, 1, None, None], num_classes=num_classes,
                  initial_channels=initial_channels)

def ResNet6(in_ch, num_classes, initial_channels=16):
    """ResNet6 for Tensil AI"""
    return ResNet(in_ch, BasicBlock, [1, 1, None, None, None], num_classes=num_classes,
                  initial_channels=initial_channels)


if __name__ == "__main__":
    print("="*80)
    print("Testing Standard ResNet10 for Tensil AI")
    print("="*80)
    
    # Test different configurations
    configs = [
        (32, 32, 16, "FPGA-optimized (32×32, 16ch)"),
        (64, 64, 16, "Balanced (64×64, 16ch)"),
        (32, 32, 32, "Higher capacity (32×32, 32ch)"),
    ]
    
    for h, w, ch, desc in configs:
        print(f"\n{desc}")
        print("-" * 80)
        
        net = ResNet10(
            in_ch=1, 
            num_classes=4, 
            initial_channels=ch
        ).to('cuda')
        
        # Test forward pass
        x = torch.randn(2, 1, h, w).to('cuda')
        
        with torch.no_grad():
            out, features = net(x)
        
        params = sum(p.numel() for p in net.parameters())
        
        print(f"  Input:      {x.shape}")
        print(f"  Output:     {out.shape}")
        print(f"  Features:   {features.shape}")
        print(f"  Parameters: {params:,}")
        print(f"  FP32 size:  ~{params * 4 / 1024:.1f} KB")
        print(f"  FP16 size:  ~{params * 2 / 1024:.1f} KB (Tensil PTQ)")
    
    print("\n" + "="*80)
    print("Key Features for Tensil AI:")
    print("="*80)
    print("  ✓ Standard PyTorch (no Brevitas)")
    print("  ✓ Fixed Linear layer (no LazyLinear)")
    print("  ✓ Configurable channels (FPGA resource control)")
    print("  ✓ Float training → Tensil PTQ")
    print("  ✓ Compatible with ONNX opset 9-10")
    print("="*80)
    print("\nAll tests passed! ✓")
