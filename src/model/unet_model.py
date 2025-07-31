"""
Advanced U-Net models for optical defect detection.
Includes standard U-Net, U-Net++, and Attention U-Net architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import math


class DoubleConv(nn.Module):
    """
    Double convolution block with batch normalization and dropout.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        mid_channels: Optional[int] = None,
        dropout_rate: float = 0.1,
        batch_norm: bool = True
    ):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class ResidualBlock(nn.Module):
    """
    Residual block for improved gradient flow.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        dropout_rate: float = 0.1,
        batch_norm: bool = True
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out


class AttentionBlock(nn.Module):
    """
    Attention block for attention U-Net.
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Down(nn.Module):
    """
    Downscaling block with maxpool and double convolution.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        dropout_rate: float = 0.1,
        batch_norm: bool = True,
        use_residual: bool = False
    ):
        super().__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate, batch_norm=batch_norm) 
            if not use_residual else ResidualBlock(in_channels, out_channels, dropout_rate, batch_norm)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block with transposed convolution and double convolution.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        bilinear: bool = True,
        dropout_rate: float = 0.1,
        batch_norm: bool = True,
        use_residual: bool = False
    ):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, 
                                 dropout_rate=dropout_rate, batch_norm=batch_norm)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate, batch_norm=batch_norm)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionUp(nn.Module):
    """
    Upscaling block with attention mechanism.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        bilinear: bool = True,
        dropout_rate: float = 0.1,
        batch_norm: bool = True
    ):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, 
                                 dropout_rate=dropout_rate, batch_norm=batch_norm)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate, batch_norm=batch_norm)
        
        self.attention = AttentionBlock(F_g=in_channels//2, F_l=out_channels, F_int=out_channels//2)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        x2 = self.attention(g=x1, x=x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output convolution layer.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture for image segmentation.
    """
    
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        initial_features: int = 64,
        depth: int = 5,
        dropout_rate: float = 0.1,
        batch_norm: bool = True,
        bilinear: bool = True,
        use_residual: bool = False
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.depth = depth
        
        # Calculate feature dimensions for each level
        self.features = [initial_features * (2 ** i) for i in range(depth)]
        
        # Initial convolution
        self.inc = DoubleConv(n_channels, self.features[0], dropout_rate=dropout_rate, batch_norm=batch_norm)
        
        # Downsampling path
        self.down_layers = nn.ModuleList()
        for i in range(depth - 1):
            self.down_layers.append(
                Down(self.features[i], self.features[i + 1], dropout_rate, batch_norm, use_residual)
            )
        
        # Bottleneck
        self.bottleneck = DoubleConv(self.features[-2], self.features[-1], dropout_rate=dropout_rate, batch_norm=batch_norm)
        
        # Upsampling path
        self.up_layers = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.up_layers.append(
                Up(self.features[i], self.features[i - 1], bilinear, dropout_rate, batch_norm, use_residual)
            )
        
        # Output convolution
        self.outc = OutConv(self.features[0], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        x1 = self.inc(x)
        features = [x1]
        
        for down in self.down_layers:
            x1 = down(x1)
            features.append(x1)
        
        # Bottleneck
        x1 = self.bottleneck(x1)
        
        # Decoder path
        for i, up in enumerate(self.up_layers):
            x1 = up(x1, features[-(i + 2)])
        
        # Output
        logits = self.outc(x1)
        return logits


class UNetPlusPlus(nn.Module):
    """
    U-Net++ architecture with dense skip connections.
    """
    
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        initial_features: int = 64,
        depth: int = 5,
        dropout_rate: float = 0.1,
        batch_norm: bool = True,
        bilinear: bool = True
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.depth = depth
        
        # Calculate feature dimensions
        self.features = [initial_features * (2 ** i) for i in range(depth)]
        
        # Initial convolution
        self.inc = DoubleConv(n_channels, self.features[0], dropout_rate=dropout_rate, batch_norm=batch_norm)
        
        # Downsampling path
        self.down_layers = nn.ModuleList()
        for i in range(depth - 1):
            self.down_layers.append(
                Down(self.features[i], self.features[i + 1], dropout_rate, batch_norm)
            )
        
        # Bottleneck
        self.bottleneck = DoubleConv(self.features[-2], self.features[-1], dropout_rate=dropout_rate, batch_norm=batch_norm)
        
        # Dense skip connections
        self.dense_blocks = nn.ModuleDict()
        for i in range(depth - 1):
            for j in range(depth - 1 - i):
                in_channels = self.features[j] + self.features[j + 1]
                self.dense_blocks[f'up_{i}_{j}'] = Up(
                    in_channels, self.features[j], bilinear, dropout_rate, batch_norm
                )
        
        # Output convolutions for each level
        self.out_convs = nn.ModuleList()
        for i in range(depth):
            self.out_convs.append(OutConv(self.features[i], n_classes))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        x1 = self.inc(x)
        encoder_features = [x1]
        
        for down in self.down_layers:
            x1 = down(x1)
            encoder_features.append(x1)
        
        # Bottleneck
        x1 = self.bottleneck(x1)
        
        # Dense decoder path
        decoder_features = [[None for _ in range(self.depth)] for _ in range(self.depth)]
        
        # Initialize decoder features
        for i in range(self.depth):
            decoder_features[i][i] = encoder_features[i]
        
        # Fill decoder features using dense connections
        for i in range(self.depth - 1):
            for j in range(self.depth - 1 - i):
                if i == 0:
                    # First level: connect from encoder
                    decoder_features[i][j] = encoder_features[j]
                else:
                    # Higher levels: dense connections
                    skip_connections = []
                    for k in range(j, j + i + 1):
                        if decoder_features[i-1][k] is not None:
                            skip_connections.append(decoder_features[i-1][k])
                    
                    if skip_connections:
                        # Upsample and concatenate
                        up_input = torch.cat(skip_connections, dim=1)
                        decoder_features[i][j] = self.dense_blocks[f'up_{i-1}_{j}'](up_input, encoder_features[j])
        
        # Output from the finest level
        output = self.out_convs[0](decoder_features[-1][0])
        return output


class AttentionUNet(nn.Module):
    """
    Attention U-Net with attention gates.
    """
    
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        initial_features: int = 64,
        depth: int = 5,
        dropout_rate: float = 0.1,
        batch_norm: bool = True,
        bilinear: bool = True
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.depth = depth
        
        # Calculate feature dimensions
        self.features = [initial_features * (2 ** i) for i in range(depth)]
        
        # Initial convolution
        self.inc = DoubleConv(n_channels, self.features[0], dropout_rate=dropout_rate, batch_norm=batch_norm)
        
        # Downsampling path
        self.down_layers = nn.ModuleList()
        for i in range(depth - 1):
            self.down_layers.append(
                Down(self.features[i], self.features[i + 1], dropout_rate, batch_norm)
            )
        
        # Bottleneck
        self.bottleneck = DoubleConv(self.features[-2], self.features[-1], dropout_rate=dropout_rate, batch_norm=batch_norm)
        
        # Upsampling path with attention
        self.up_layers = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.up_layers.append(
                AttentionUp(self.features[i], self.features[i - 1], bilinear, dropout_rate, batch_norm)
            )
        
        # Output convolution
        self.outc = OutConv(self.features[0], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        x1 = self.inc(x)
        features = [x1]
        
        for down in self.down_layers:
            x1 = down(x1)
            features.append(x1)
        
        # Bottleneck
        x1 = self.bottleneck(x1)
        
        # Decoder path with attention
        for i, up in enumerate(self.up_layers):
            x1 = up(x1, features[-(i + 2)])
        
        # Output
        logits = self.outc(x1)
        return logits


class DeepSupervisionUNet(nn.Module):
    """
    U-Net with deep supervision for multi-scale predictions.
    """
    
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        initial_features: int = 64,
        depth: int = 5,
        dropout_rate: float = 0.1,
        batch_norm: bool = True,
        bilinear: bool = True
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.depth = depth
        
        # Calculate feature dimensions
        self.features = [initial_features * (2 ** i) for i in range(depth)]
        
        # Initial convolution
        self.inc = DoubleConv(n_channels, self.features[0], dropout_rate=dropout_rate, batch_norm=batch_norm)
        
        # Downsampling path
        self.down_layers = nn.ModuleList()
        for i in range(depth - 1):
            self.down_layers.append(
                Down(self.features[i], self.features[i + 1], dropout_rate, batch_norm)
            )
        
        # Bottleneck
        self.bottleneck = DoubleConv(self.features[-2], self.features[-1], dropout_rate=dropout_rate, batch_norm=batch_norm)
        
        # Upsampling path
        self.up_layers = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.up_layers.append(
                Up(self.features[i], self.features[i - 1], bilinear, dropout_rate, batch_norm)
            )
        
        # Output convolutions for deep supervision
        self.out_convs = nn.ModuleList()
        for i in range(depth):
            self.out_convs.append(OutConv(self.features[i], n_classes))
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Encoder path
        x1 = self.inc(x)
        features = [x1]
        
        for down in self.down_layers:
            x1 = down(x1)
            features.append(x1)
        
        # Bottleneck
        x1 = self.bottleneck(x1)
        
        # Decoder path with deep supervision
        outputs = []
        for i, up in enumerate(self.up_layers):
            x1 = up(x1, features[-(i + 2)])
            outputs.append(self.out_convs[i](x1))
        
        # Final output
        final_output = self.out_convs[-1](x1)
        outputs.append(final_output)
        
        return outputs


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params
    }


def create_model(
    model_type: str = "unet",
    n_channels: int = 3,
    n_classes: int = 1,
    initial_features: int = 64,
    depth: int = 5,
    dropout_rate: float = 0.1,
    batch_norm: bool = True,
    bilinear: bool = True
) -> nn.Module:
    """
    Create a model based on the specified type.
    
    Args:
        model_type: Type of model ('unet', 'unetpp', 'attention_unet', 'deep_supervision')
        n_channels: Number of input channels
        n_classes: Number of output classes
        initial_features: Number of initial features
        depth: Depth of the network
        dropout_rate: Dropout rate
        batch_norm: Whether to use batch normalization
        bilinear: Whether to use bilinear upsampling
        
    Returns:
        PyTorch model
    """
    if model_type == "unet":
        return UNet(n_channels, n_classes, initial_features, depth, dropout_rate, batch_norm, bilinear)
    elif model_type == "unetpp":
        return UNetPlusPlus(n_channels, n_classes, initial_features, depth, dropout_rate, batch_norm, bilinear)
    elif model_type == "attention_unet":
        return AttentionUNet(n_channels, n_classes, initial_features, depth, dropout_rate, batch_norm, bilinear)
    elif model_type == "deep_supervision":
        return DeepSupervisionUNet(n_channels, n_classes, initial_features, depth, dropout_rate, batch_norm, bilinear)
    else:
        raise ValueError(f"Unknown model type: {model_type}")