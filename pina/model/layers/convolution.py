"""Module for Base Continuous Convolution class."""

from abc import ABCMeta, abstractmethod
import torch


class BaseContinuousConv(torch.nn.Module, metaclass=ABCMeta):
    """
    Abstract class
    """

    def __init__(self, input_numb_field, output_numb_field,
                 filter_dim, stride, model=None):
        """Base Class for Continuous Convolution.

        The algorithm expects input to be in the form:
        $$[B \times N_{in} \times N \times D]$$
        where $B$ is the batch_size, $N_{in}$ is the number of input
        fields, $N$ the number of points in the mesh, $D$ the dimension
        of the problem.

        The algorithm returns a tensor of shape:
        $$[B \times N_{out} \times N' \times D]$$
        where $B$ is the batch_size, $N_{out}$ is the number of output
        fields, $N'$ the number of points in the mesh, $D$ the dimension
        of the problem.

        :param input_numb_field: number of fields in the input
        :type input_numb_field: int
        :param output_numb_field: number of fields in the output
        :type output_numb_field: int
        :param filter_dim: dimension of the filter
        :type filter_dim: tuple/ list
        :param stride: stride for the filter
        :type stride: dict
        :param model: neural network for inner parametrization,
        defaults to None
        :type model: torch.nn.Module, optional
        """
        super().__init__()

        if isinstance(input_numb_field, int):
            self._input_numb_field = input_numb_field
        else:
            raise ValueError('input_numb_field must be int.')

        if isinstance(output_numb_field, int):
            self._output_numb_field = output_numb_field
        else:
            raise ValueError('input_numb_field must be int.')

        if isinstance(filter_dim, (tuple, list)):
            vect = filter_dim
        else:
            raise ValueError('filter_dim must be tuple or list.')
        vect = torch.tensor(vect)
        self.register_buffer("_dim", vect, persistent=False)

        if isinstance(stride, dict):
            self._stride = stride
        else:
            raise ValueError('stride must be dictionary.')

        self._net = model

    @ property
    def net(self):
        return self._net

    @ property
    def stride(self):
        return self._stride

    @ property
    def dim(self):
        return self._dim

    @ property
    def input_numb_field(self):
        return self._input_numb_field

    @ property
    def output_numb_field(self):
        return self._output_numb_field

    @property
    @abstractmethod
    def forward(self, x):
        pass
