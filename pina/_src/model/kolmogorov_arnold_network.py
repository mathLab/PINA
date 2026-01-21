"""Kolmogorov Arnold Network implementation"""
import torch
import torch.nn as nn
from typing import List

from pina._src.model.block.kan_block import KANBlock

class KolmogorovArnoldNetwork(torch.nn.Module):
    """
    Kolmogorov Arnold Network, a neural network using KAN layers instead of
    traditional MLP layers. Each layer uses learnable univariate functions
    (B-splines + base functions) on edges.

    .. references::

        Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M.,
        ... & Tegmark, M. (2024). Kan: Kolmogorov-arnold networks. arXiv
        preprint arXiv:2404.19756.

    """
    
    def __init__(
        self, 
        layer_sizes: List[int],
        k: int = 3,
        num: int = 3,
        grid_eps: float = 0.1,
        grid_range: List[float] = [-1, 1],
        grid_extension: bool = True,
        noise_scale: float = 0.1,
        base_function = torch.nn.SiLU(),
        scale_base_mu: float = 0.0,
        scale_base_sigma: float = 1.0,
        scale_sp: float = 1.0,
        inner_nodes: int = 5,
        sparse_init: bool = False,
        sp_trainable: bool = True,
        sb_trainable: bool = True,
        save_act: bool = True
    ):
        """
        Initialize the KAN network.

        :param iterable layer_sizes: List of layer sizes including input and
            output dimensions.
        :param int k: Order of the B-spline.
        :param int num: Number of grid points for B-splines.
        :param float grid_eps: Epsilon for grid spacing.
        :param list grid_range: Range for the grid [min, max].
        :param bool grid_extension: Whether to extend the grid.
        :param float noise_scale: Scale for initialization noise.
        :param base_function: Base activation function (e.g., SiLU).
        :param float scale_base_mu: Mean for base function scaling.
        :param float scale_base_sigma: Std for base function scaling.
        :param float scale_sp: Scale for spline functions.
        :param int inner_nodes: Number of inner nodes for KAN layers.
        :param bool sparse_init: Whether to use sparse initialization.
        :param bool sp_trainable: Whether spline parameters are trainable.
        :param bool sb_trainable: Whether base function parameters are
            trainable.
        :param bool save_act: Whether to save activations after each layer.
        """
        super().__init__()
        
        if len(layer_sizes) < 2:
            raise ValueError("Need at least input and output dimensions")
        
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.save_act = save_act
        
        # Create KAN layers
        self.kan_layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            layer = KANBlock(
                k=k,
                input_dimensions=layer_sizes[i],
                output_dimensions=layer_sizes[i+1],
                num=num,
                grid_eps=grid_eps,
                grid_range=grid_range,
                grid_extension=grid_extension,
                noise_scale=noise_scale,
                base_function=base_function,
                scale_base_mu=scale_base_mu,
                scale_base_sigma=scale_base_sigma,
                scale_sp=scale_sp,
                inner_nodes=inner_nodes,
                sparse_init=sparse_init,
                sp_trainable=sp_trainable,
                sb_trainable=sb_trainable
            )
            self.kan_layers.append(layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the KAN network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dimensions)
            
        Returns:
            Output tensor of shape (batch_size, output_dimensions)
        """
        current = x
        self.acts = [current]

        for i, layer in enumerate(self.kan_layers):
            current = layer(current)
            # current = torch.nn.functional.sigmoid(current)
            
            if self.save_act:
                self.acts.append(current.detach())
            
        return current
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
    def update_grid_from_samples(self, x: torch.Tensor, mode: str = 'sample'):
        """
        Update grid for all layers based on input samples.
        This adapts the grid points to better fit the data distribution.
        
        Args:
            x: Input samples, shape (batch_size, input_dimensions)
            mode: 'sample' or 'grid' - determines sampling strategy
        """
        current = x
        
        for i, layer in enumerate(self.kan_layers):
            layer.update_grid_from_samples(current, mode=mode)
            
            if i < len(self.kan_layers) - 1:
                with torch.no_grad():
                    current = layer(current)
                    
    def update_grid_resolution(self, new_num: int):
        """
        Update the grid resolution for all layers.
        This can be used for adaptive training where grid resolution increases over time.
        
        Args:
            new_num: New number of grid points
        """
        for layer in self.kan_layers:
            layer.update_grid_resolution(new_num)
            
    def enable_sparsification(self, threshold: float = 1e-4):
        """
        Enable sparsification by setting small weights to zero.
        
        Args:
            threshold: Threshold below which weights are set to zero
        """
        with torch.no_grad():
            for layer in self.kan_layers:
                # Sparsify scale parameters
                layer.scale_base.data[torch.abs(layer.scale_base.data) < threshold] = 0
                layer.scale_spline.data[torch.abs(layer.scale_spline.data) < threshold] = 0
                
                # Update mask
                layer.mask.data = ((torch.abs(layer.scale_base) >= threshold) | 
                             (torch.abs(layer.scale_spline) >= threshold)).float()

    def get_activation_statistics(self, x: torch.Tensor):
        """
        Get statistics about activations for analysis purposes.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with activation statistics
        """
        stats = {}
        current = x
        
        for i, layer in enumerate(self.kan_layers):
            current = layer(current)
            stats[f'layer_{i}'] = {
                'mean': current.mean().item(),
                'std': current.std().item(),
                'min': current.min().item(),
                'max': current.max().item()
            }
            
        return stats
    
    
    def get_network_grid_statistics(self):
        """
        Get grid statistics for all layers in the network.
        
        Returns:
            Dictionary with grid statistics for each layer
        """
        stats = {}
        for i, layer in enumerate(self.kan_layers):
            stats[f'layer_{i}'] = layer.get_grid_statistics()
        return stats

  