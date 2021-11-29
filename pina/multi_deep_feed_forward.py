
from .problem import Problem
import torch
import torch.nn as nn
import numpy as np
from .cube import Cube
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from torch.nn import Tanh, ReLU
import torch.nn.utils.prune as prune
from pina.adaptive_functions import AdaptiveLinear
from pina.deep_feed_forward import DeepFeedForward

class MultiDeepFeedForward(torch.nn.Module):

    def __init__(self, dff_dict):
        '''
        '''
        super().__init__()

        if not isinstance(dff_dict, dict):
            raise TypeError

        for name, constructor_args in dff_dict.items():
            setattr(self, name, DeepFeedForward(**constructor_args))
        
        

