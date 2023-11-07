import torch
import torch.nn as nn
from ..utils import check_consistency


class Network(torch.nn.Module):
    """
    Network class with standard forward method 
    and possibility to pass extra features. This
    class is used internally in PINA to convert
    any :class:`torch.nn.Module`s in a PINA module.
    """

    def __init__(self, model, extra_features=None):
        super().__init__()

        # check model consistency
        check_consistency(model, nn.Module)
        self._model = model

        # check consistency and assign extra fatures
        if extra_features is None:
            self._extra_features = []
        else:
            for feat in extra_features:
                check_consistency(feat, nn.Module)
            self._extra_features = nn.Sequential(*extra_features)

        # check model works with inputs
        # TODO

    def forward(self, x):
        """
        Forward method for Network class. This class
        implements the standard forward method, and
        it adds the possibility to pass extra features.

        :param torch.Tensor x: Input of the network.
        :return torch.Tensor: Output of the network.
        """
        # extract features and append
        for feature in self._extra_features:
            x = x.append(feature(x))
        # perform forward pass
        return self._model(x)

    @property
    def model(self):
        return self._model

    @property
    def extra_features(self):
        return self._extra_features
