import torch
import torch.nn as nn
from ..utils import check_consistency
from ..label_tensor import LabelTensor


class Network(torch.nn.Module):

    def __init__(
        self, model, input_variables, output_variables, extra_features=None
    ):
        """
        Network class with standard forward method
        and possibility to pass extra features. This
        class is used internally in PINA to convert
        any :class:`torch.nn.Module` s in a PINA module.

        :param model: The torch model to convert in a PINA model.
        :type model: torch.nn.Module
        :param list(str) input_variables: The input variables of the :class:`AbstractProblem`, whose type depends on the
            type of domain (spatial, temporal, and parameter).
        :param list(str) output_variables: The output variables of the :class:`AbstractProblem`, whose type depends on the
            problem setting.
        :param extra_features: List of torch models to augment the input, defaults to None.
        :type extra_features: list(torch.nn.Module)
        """
        super().__init__()

        # check model consistency
        check_consistency(model, nn.Module)
        check_consistency(input_variables, str)
        if output_variables is not None:
            check_consistency(output_variables, str)

        self._model = model
        self._input_variables = input_variables
        self._output_variables = output_variables

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
        # only labeltensors as input
        assert isinstance(
            x, LabelTensor
        ), "Expected LabelTensor as input to the model."

        # extract torch.Tensor from corresponding label
        # in case `input_variables = []` all points are used
        if self._input_variables:
            x = x.extract(self._input_variables)
        # extract features and append
        for feature in self._extra_features:
            x = x.append(feature(x))

        # perform forward pass + converting to LabelTensor

        output = self._model(x.as_subclass(torch.Tensor))
        if self._output_variables is not None:
            output = LabelTensor(output, self._output_variables)

        return output

    # TODO to remove in next releases (only used in GAROM solver)
    def forward_map(self, x):
        """
        Forward method for Network class when the input is
        a tuple. This class is simply a forward with the input casted as a
        tuple or list :class`torch.Tensor`.
        All the PINA models ``forward`` s are overriden
        by this class, to enable :class:`pina.label_tensor.LabelTensor` labels
        extraction.

        :param list (torch.Tensor) | tuple(torch.Tensor) x: Input of the network.
        :return torch.Tensor: Output of the network.

        .. note::
            This function does not extract the input variables, all the variables
            are used for both tensors. Output variables are correctly applied.
        """

        # perform forward pass (using torch.Tensor) + converting to LabelTensor
        output = LabelTensor(self._model(x.tensor), self._output_variables)
        return output

    @property
    def torchmodel(self):
        return self._model

    @property
    def extra_features(self):
        return self._extra_features
