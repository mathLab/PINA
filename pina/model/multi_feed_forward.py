import torch

from .feed_forward import FeedForward


class MultiFeedForward(torch.nn.Module):

    def __init__(self, dff_dict):
        '''
        '''
        super().__init__()

        if not isinstance(dff_dict, dict):
            raise TypeError

        for name, constructor_args in dff_dict.items():
            setattr(self, name, FeedForward(**constructor_args))
