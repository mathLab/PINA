""" Module for Averaging Neural Operator Layer class. """

from torch import nn, mean
from torch_geometric.nn import MessagePassing, InstanceNorm, radius_graph

from pina.utils import check_consistency


class MessagePassingBlock(nn.Module):


