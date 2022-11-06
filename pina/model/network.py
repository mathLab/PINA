import torch.nn as nn
import torch
from pina.label_tensor import LabelTensor


class Network(nn.Module):
    """The PINA implementation of any neural network.

    :param iterable(torch.nn.Module) model: the torch model of the network
    :param list(str) input_variables: the list containing the labels
        corresponding to the input components of the model.
    :param list(str) output_variables: the list containing the labels
        corresponding to the components of the output computed by the model.
    :param iterable(torch.nn.Module) extra_features: the additional input
        features to use as augmented input.

    :Example:
            >>> class SimpleNet(nn.Module):
            >>>    def __init__(self):
            >>>        super().__init__()
            >>>        self.layers = nn.Sequential(
            >>>        nn.Linear(3, 20),
            >>>        nn.Tanh(),
            >>>        nn.Linear(20, 1)
            >>>        )
            >>>    def forward(self, x):
            >>>        return self.layers(x)
            >>> net = SimpleNet()
            >>> input_variables = ['x', 'y']
            >>> output_variables =['u']
            >>> model_feat = Network(net, input_variables, output_variables)
            Network(
                (extra_features): Sequential()
                (model): Sequential(
                    (0): Linear(in_features=2, out_features=20, bias=True)
                    (1): Tanh()
                    (2): Linear(in_features=20, out_features=1, bias=True)
                )
            )
    """

    def __init__(self, model, input_variables,
                 output_variables, extra_features=None):
        super().__init__()

        if extra_features is None:
            extra_features = []

        self._extra_features = nn.Sequential(*extra_features)
        self._model = model
        self._input_variables = input_variables
        self._output_variables = output_variables

        try:
            tmp = torch.rand((10, len(input_variables)))
            self._model(tmp)
        except:
            raise ValueError('Error in constructing the PINA network.'
                             ' Check compatibility of input/output'
                             ' variables shape with the torch model'
                             ' or check the correctness of the torch'
                             ' model itself.')

    def forward(self, x):
        """Forward method for Network class

        :param torch.tensor x: input of the network
        :return torch.tensor: output of the network
        """

        x = x.extract(self._input_variables)

        for feature in self._extra_features:
            x = x.append(feature(x))

        output = self._model(x).as_subclass(LabelTensor)
        output.labels = self._output_variables

        return output

    @property
    def input_variables(self):
        return self._input_variables

    @property
    def output_variables(self):
        return self._output_variables

    @property
    def extra_features(self):
        return self._extra_features

    @property
    def model(self):
        return self._model
