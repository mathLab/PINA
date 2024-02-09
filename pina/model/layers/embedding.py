import torch
from pina.utils import check_consistency


class FourierEmbedding(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, period,
                 func=torch.nn.Identity, trainable_params=False):
        super().__init__()

        # check input consistency
        check_consistency(period, (float, int))
        check_consistency(func, torch.nn.Module, subclass=True)
        check_consistency(trainable_params, bool)
        check_consistency(input_dimension, int)
        check_consistency(output_dimension, int)

        # define the parameters
        self._activation = func
        one_ = torch.tensor([1.])
        zero_ = torch.tensor([0.])
        self._A = torch.nn.Parameter(one_) if trainable_params else one_
        self._phi = torch.nn.Parameter(zero_) if trainable_params else zero_
        self._c = torch.nn.Parameter(zero_) if trainable_params else zero_
        self._omega = 2 * torch.pi / period

    def forward(self, x):
        return self._A * torch.cos(self._omega * x + self._phi) + self._c
    
    @property
    def period(self):
        return 2 * torch.pi / self._omega
    
    @property
    def func(self):
        return self._activation
    

f = torch.nn.Sequential(InfinitePeriodic1D(),
                        torch.nn.Linear())