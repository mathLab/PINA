import torch
import torch.nn as nn
import math

# used copilot for this, learn what the periods do here
from .message_passing.en_equivariant_network_block import (
    EnEquivariantNetworkBlock,
)
from .temporal_convolution_layer import TemporalConvolutionLayer

def time_embeddings(timesteps, embedding_dim):
    '''
    Will create faster vectorized version
    '''
    zeros = torch.zeros(timesteps, embedding_dim)
    half_dim = embedding_dim // 2
    denominator = half_dim - 1

    for i in range(timesteps):
        for j in range(half_dim):
            zeros[i, 2*j] = math.sin(i / (10000 ** (j/ denominator)))
            zeros[i, (2*j + 1)] = math.cos(i / (10000 ** (j/ denominator)))
    return zeros

class EquivariantGraphNeuralOperatorBlock(nn.Module):
    def __init__(
        self,
        time_discretizations,
        num_scalar_features,
        num_vec_features,
        embedding_dim,
        n_nodes,
        f_z_size,
        n_modes,
        eq_activation,
        scalar_feature_activation,
    ):
        super().__init__()
    
        # determine f_h size from time embeddding dim and num_scalar features
        f_h_size = num_scalar_features + embedding_dim
        f_z_size = num_vec_features
        
        self.conv_layer = TemporalConvolutionLayer(
            time_discretizations, n_nodes,
            f_h_size, f_z_size, n_modes, eq_activation,
            scalar_feature_activation)
        
        self._time_discretizations = time_discretizations
        self._n_nodes = n_nodes
        
        #! Need to figure out the correct parameters given the arguments to EGNO BLock
        self.egnn_layer = EnEquivariantNetworkBlock(f_h_size, edges?, f_h_size)

    def forward(
        self, features, positions, edge_indicies, edge_attributes, mean
    ):
        """
        features: tensor [Nodes, H]
        positions: tensor [Nodes, 3 (For coordinates)]
        edge_indicies: tensor [Edges, 2]
        edge_attributes: tensor [Edges, Features]


        Assuming time discretization has been done in the overall model before running any block.

        Should be able to gather the data from the graph itself in the overall model file

        Mean for CoM caculation is only calculated once at the beginning

        Potentailly don't need split in temporal_conv_layer if input separate coordinate and feature inputs
        """
        # time discretize
        #! Need to make my own version of this
        #! My attempt on local
        # Arguments -> [Time_discretizations, embedding_dim]
        time_embds = time_embeddings(self._time_discretizations)

        # may want to use expand() to avoid copy of tensor (would need to sandwich operations)
        # [Nodes, Scalar features] ->[Time_discretizations, Nodes, Features]
        features = features.repeat(self._time_discretizations, 1, 1)

        # Repeats the embedding of the time for each node
        # (every node in each Time_discretization has same time_embed)
        # [Time_discretizations, embedding_dim] -> [Time_discretizations, Nodes, embedding_dim]
        time_embds = time_embds.repeat(1, self._n_nodes, 1)

        # Features with time integrated (Adds the embedding vector to the end of each feature vector)
        # -> [Time_discretizatinos, Nodes, features + embedding_dim]
        features_with_time = torch.cat((features, time_embds), dim=2)

        #? # flattening for layers
        #? flattened = features.view(-1, features_with_time.shape[-1])


        #? # does repeating and flattening together
        #? # add another 3d vec features below in the same way
        #? # [Nodes, Coords] -> [Time_discretizations * Nodes, Coords]
        #? positions = positions.repeat(self._time_discretizations, 1)

        # [Nodes, Coords] -> [Time_discretizations, Nodes, Coords]
        positions = positions.repeat(self._time_discretizations, 1, 1)

        # mean used to cancel CoM extended over all time
        # [Time_discretizations, 1]
        #! This is broadcastable by subtracting with postions as shown below:
        #! Is this mean over the entire graph and could it be represented as an integer
        #! Right now assuming that mean input is [Nodes, Mean]
        # Positions: [Time_discretizations * Nodes, Coords]
        # Mean     : [Time_discretizations, 1]
        # mean = mean.repeat(self._time_discretizations, 1)

        #! Right now assuming that mean input is [Nodes, Mean]
        mean = mean.repeat(self._time_discretizations, 1, 1)

        # Adjust Positions with mean here
        positions = positions - mean

        # Apply temporal conv layer
        features_temp_conv, positions_temp_conv = self.conv_layer(features_with_time, positions)

        # Add back in mean
        positions_temp_conv = positions_temp_conv + mean

        # Apply EGNN layer
        features, positions = self.egnn_layer(features_temp_conv.view(self._time_discretizations * self._n_nodes, -1),
                        positions_temp_conv.view(self._time_discretizations * self._n_nodes, -1),
                        edge_indicies,
                        edge_attributes)
        
        return features, positions