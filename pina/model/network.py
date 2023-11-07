import torch
import torch.nn as nn
from ..utils import check_consistency


class Network(torch.nn.Module):

    def __init__(self, model, extra_features=None):
        """
        Network class with standard forward method
        and possibility to pass extra features. This
        class is used internally in PINA to convert
        any :class:`torch.nn.Module` s in a PINA module.
    
        :param model: The torch model to convert in a PINA model.
        :type model: torch.nn.Module
        :param extra_features: List of torch models to augment the input, defaults to None.
        :type extra_features: list(torch.nn.Module)
        """
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
        All the PINA models ``forward`` s are overriden
        by this class, to enable :class:`pina.label_tensor.LabelTensor` labels
        extraction.

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
