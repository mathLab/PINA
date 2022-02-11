import torch
import torch.nn as nn

from pina.label_tensor import LabelTensor


class FeedForward(torch.nn.Module):

    def __init__(self, input_variables, output_variables, inner_size=20,
                 n_layers=2, func=nn.Tanh, layers=None, extra_features=None):
        '''
        '''
        super().__init__()

        if extra_features is None:
            extra_features = []
        self.extra_features = nn.Sequential(*extra_features)

        self.input_variables = input_variables
        self.input_dimension = len(input_variables)

        self.output_variables = output_variables
        self.output_dimension = len(output_variables)

        n_features = len(extra_features)

        if layers is None:
            layers = [inner_size] * n_layers

        tmp_layers = layers.copy()
        tmp_layers.insert(0, self.input_dimension+n_features)
        tmp_layers.append(self.output_dimension)

        self.layers = []
        for i in range(len(tmp_layers)-1):
            self.layers.append(nn.Linear(tmp_layers[i], tmp_layers[i+1]))

        if isinstance(func, list):
            self.functions = func
        else:
            self.functions = [func for _ in range(len(self.layers)-1)]

        unique_list = []
        for layer, func in zip(self.layers[:-1], self.functions):
            unique_list.append(layer)
            if func is not None:
                unique_list.append(func())
        unique_list.append(self.layers[-1])

        self.model = nn.Sequential(*unique_list)

    def forward(self, x):
        """
        """
        nf = len(self.extra_features)
        if nf == 0:
            return LabelTensor(self.model(x.tensor), self.output_variables)

        # if self.extra_features
        input_ = torch.zeros(x.shape[0], nf+x.shape[1], dtype=x.dtype,
                             device=x.device)
        input_[:, :x.shape[1]] = x.tensor
        for i, feature in enumerate(self.extra_features,
                                    start=self.input_dimension):
            input_[:, i] = feature(x)
        return LabelTensor(self.model(input_), self.output_variables)
