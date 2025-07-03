import torch
import torch.nn as nn
# used copilot for this, learn what the periods do here
from .message_passing.en_equivariant_network_block import EnEquivariantNetworkBlock
from .temporal_convolution_layer import TemporalConvolutionLayer


class EquivariantGraphNeuralOperatorBlock(nn.Module):
    def __init__(self, time_discretizations, n_nodes, f_h_size, f_z_size, n_modes, eq_actiation, scalar_feature_activation):
        super().__init__()
        
    #! edge_index vs edge_attr?
    def forward(self, x, pos, edge_index, edge_attr, mean):
        '''
        Assuming time discretization has been done in the overall model before running any block.

        Should be able to gather the data from the graph itself in the overall model file

        Mean for CoM caculation is only calculated once at the beginning

        Potentailly don't need split in temporal_conv_layer if input separate coordinate and feature inputs
        '''
        # Cancel center of mass with 1/N\sum_{i=1}^N x_i
        # Pass into TemporalConvLayer
        x = TemporalConvolutionLayer(x)
        
        # Reshape tensor so that EGNN layers will only operate on the node and channel dimension, while treating the temporal dimension indentical to a batch dimension

        # Apply EGNN layer

        # Return





