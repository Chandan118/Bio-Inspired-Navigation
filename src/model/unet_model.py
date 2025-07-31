"""
Advanced U-Net model for optical diagnostics defect detection.
Implements attention gates, residual connections, and multi-scale feature fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any
import math
import logging

logger = logging.getLogger(__name__)

class AttentionGate(nn.Module):
    """
    Attention gate for U-Net architecture.
    Helps the model focus on relevant features during upsampling.
    """
    
    def __init__(self, in_channels: int, gating_channels: int, inter_channels: int = None):
        super(AttentionGate, self).__init__()
        
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels or in_channels // 2
        
        # Linear transformations
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi = nn.Conv2d(gating_channels, self.inter_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.psi = nn.Conv2d(self.inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Batch normalization
        self.bn_theta = nn.BatchNorm2d(self.inter_channels)
        self.bn_phi = nn.BatchNorm2d(self.inter_channels)
        self.bn_psi = nn.BatchNorm2d(1)
        
        # Final convolution
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of attention gate.
        
        Args:
            x: Input feature map (skip connection)
            g: Gating signal (from encoder)
        
        Returns:
            Attention-weighted feature map
        """
        batch_size, c, h, w = x.size()
        
        # Transform inputs
        theta_x = self.bn_theta(self.theta(x))
        phi_g = self.bn_phi(self.phi(g))
        
        # Resize gating signal to match x dimensions
        if phi_g.size(2) != h or phi_g.size(3) != w:
            phi_g = F.interpolate(phi_g, size=(h, w), mode='bilinear', align_corners=False)
        
        # Compute attention weights
        f = F.relu(theta_x + phi_g, inplace=True)
        f = self.bn_psi(self.psi(f))
        f = torch.sigmoid(f)
        
        # Apply attention
        y = x * f
        
        # Final transformation
        y = self.final_conv(y)
        
        return y

class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization and activation.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout_rate: float = 0.1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of residual block."""
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class DoubleConv(nn.Module):
    """
    Double convolution block with optional residual connections.
    """
    
    def __init__(self, in_channels: int, out_channels: int, residual: bool = True, dropout_rate: float = 0.1):
        super(DoubleConv, self).__init__()
        
        self.residual = residual
        
        if residual:
            self.conv = nn.Sequential(
                ResidualBlock(in_channels, out_channels, dropout_rate=dropout_rate),
                ResidualBlock(out_channels, out_channels, dropout_rate=dropout_rate)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of double convolution block."""
        return self.conv(x)

class EncoderBlock(nn.Module):
    """
    Encoder block with downsampling and feature extraction.
    """
    
    def __init__(self, in_channels: int, out_channels: int, residual: bool = True, dropout_rate: float = 0.1):
        super(EncoderBlock, self).__init__()
        
        self.double_conv = DoubleConv(in_channels, out_channels, residual, dropout_rate)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of encoder block.
        
        Returns:
            Tuple of (pooled_features, skip_connection)
        """
        features = self.double_conv(x)
        pooled = self.pool(features)
        
        return pooled, features

class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling and attention gates.
    """
    
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int, 
                 attention: bool = True, residual: bool = True, dropout_rate: float = 0.1):
        super(DecoderBlock, self).__init__()
        
        self.attention = attention
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        if attention:
            self.attention_gate = AttentionGate(skip_channels, in_channels)
            self.conv = DoubleConv(out_channels + skip_channels, out_channels, residual, dropout_rate)
        else:
            self.conv = DoubleConv(out_channels + skip_channels, out_channels, residual, dropout_rate)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of decoder block.
        
        Args:
            x: Input features from previous decoder block
            skip: Skip connection from encoder
        
        Returns:
            Upsampled and processed features
        """
        x = self.up(x)
        
        # Handle size mismatch
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=False)
        
        if self.attention:
            skip = self.attention_gate(skip, x)
        
        # Concatenate features
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        
        return x

class UNet(nn.Module):
    """
    Advanced U-Net architecture for optical diagnostics defect detection.
    
    Features:
    - Attention gates for better feature selection
    - Residual connections for improved gradient flow
    - Multi-scale feature fusion
    - Configurable depth and feature dimensions
    - Dropout for regularization
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 1,
        initial_features: int = 64,
        depth: int = 5,
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        attention_gates: bool = True,
        residual_connections: bool = True
    ):
        super(UNet, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.initial_features = initial_features
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.attention_gates = attention_gates
        self.residual_connections = residual_connections
        
        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        self.skip_connections = []
        
        in_channels = input_channels
        for i in range(depth):
            out_channels = initial_features * (2 ** i)
            self.encoder_blocks.append(
                EncoderBlock(in_channels, out_channels, residual_connections, dropout_rate)
            )
            in_channels = out_channels
        
        # Bottleneck
        bottleneck_channels = initial_features * (2 ** depth)
        self.bottleneck = DoubleConv(in_channels, bottleneck_channels, residual_connections, dropout_rate)
        
        # Decoder path
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            in_channels = bottleneck_channels if i == depth - 1 else initial_features * (2 ** (i + 1))
            out_channels = initial_features * (2 ** i)
            skip_channels = initial_features * (2 ** i)
            
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels, out_channels, skip_channels,
                    attention_gates, residual_connections, dropout_rate
                )
            )
        
        # Final convolution
        self.final_conv = nn.Conv2d(initial_features, output_channels, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"UNet initialized with depth {depth}, initial features {initial_features}")
        logger.info(f"Attention gates: {attention_gates}, Residual connections: {residual_connections}")
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of U-Net.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Output tensor of shape (batch_size, output_channels, height, width)
        """
        # Encoder path
        skip_connections = []
        
        for encoder_block in self.encoder_blocks:
            x, skip = encoder_block(x)
            skip_connections.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = skip_connections[-(i + 1)]
            x = decoder_block(x, skip)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate feature maps for analysis.
        
        Args:
            x: Input tensor
        
        Returns:
            Dictionary of feature maps at different scales
        """
        feature_maps = {}
        
        # Encoder features
        for i, encoder_block in enumerate(self.encoder_blocks):
            x, skip = encoder_block(x)
            feature_maps[f'encoder_{i}'] = skip
        
        # Bottleneck features
        x = self.bottleneck(x)
        feature_maps['bottleneck'] = x
        
        # Decoder features
        skip_connections = list(feature_maps.values())[:-1]  # Exclude bottleneck
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = skip_connections[-(i + 1)]
            x = decoder_block(x, skip)
            feature_maps[f'decoder_{i}'] = x
        
        return feature_maps
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get model size in megabytes."""
        param_size = 0
        buffer_size = 0
        
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

class MultiScaleUNet(nn.Module):
    """
    Multi-scale U-Net with feature pyramid network for better defect detection.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 1,
        initial_features: int = 64,
        depth: int = 5,
        scales: List[float] = [1.0, 0.5, 0.25],
        dropout_rate: float = 0.2
    ):
        super(MultiScaleUNet, self).__init__()
        
        self.scales = scales
        
        # Create U-Net for each scale
        self.unets = nn.ModuleList([
            UNet(input_channels, output_channels, initial_features, depth, dropout_rate)
            for _ in scales
        ])
        
        # Feature fusion
        total_features = len(scales) * output_channels
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_features, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale processing."""
        outputs = []
        
        for i, (scale, unet) in enumerate(zip(self.scales, self.unets)):
            if scale != 1.0:
                # Resize input
                h, w = x.size(2), x.size(3)
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            else:
                scaled_x = x
            
            # Process with U-Net
            output = unet(scaled_x)
            
            # Resize back to original size
            if scale != 1.0:
                output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
            
            outputs.append(output)
        
        # Concatenate and fuse
        fused = torch.cat(outputs, dim=1)
        final_output = self.fusion_conv(fused)
        
        return final_output

def create_unet_model(config) -> UNet:
    """Create U-Net model from configuration."""
    return UNet(
        input_channels=config.model.input_channels,
        output_channels=config.model.output_channels,
        initial_features=config.model.initial_features,
        depth=config.model.depth,
        dropout_rate=config.model.dropout_rate,
        batch_norm=config.model.batch_norm,
        attention_gates=config.model.attention_gates,
        residual_connections=config.model.residual_connections
    )

def create_multiscale_unet_model(config) -> MultiScaleUNet:
    """Create multi-scale U-Net model from configuration."""
    return MultiScaleUNet(
        input_channels=config.model.input_channels,
        output_channels=config.model.output_channels,
        initial_features=config.model.initial_features,
        depth=config.model.depth,
        dropout_rate=config.model.dropout_rate
    )