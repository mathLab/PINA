"""Module for Base Continuous Convolution class."""
from abc import ABCMeta, abstractmethod
import torch
from .stride import Stride
from .utils_convolution import optimizing


class BaseContinuousConv(torch.nn.Module, metaclass=ABCMeta):
    """
    Abstract class
    """

    def __init__(self, input_numb_field, output_numb_field,
                 filter_dim, stride, model=None, optimize=False,
                 no_overlap=False):
        """Base Class for Continuous Convolution.

        The algorithm expects input to be in the form:
        $$[B \times N_{in} \times N \times D]$$
        where $B$ is the batch_size, $N_{in}$ is the number of input
        fields, $N$ the number of points in the mesh, $D$ the dimension
        of the problem. In particular:
        * $D$ is the number of spatial variables + 1. The last column must
            contain the field value. For example for 2D problems $D=3$ and
            the tensor will be something like `[first coordinate, second
            coordinate, field value]`.
        * $N_{in}$ represents the number of vectorial function presented.
            For example a vectorial function $f = [f_1, f_2]$ will have
            $N_{in}=2$.

        :Note
            A 2-dimensional vectorial function $N_{in}=2$ of 3-dimensional
            input $D=3+1=4$ with 100 points input mesh and batch size of 8
            is represented as a tensor `[8, 2, 100, 4]`, where the columns
            `[:, 0, :, -1]` and `[:, 1, :, -1]` represent the first and
            second filed value respectively

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
        :param optimize: flag for performing optimization on the continuous
            filter, defaults to False. The flag `optimize=True` should be
            used only when the scatter datapoints are fixed through the
            training. If torch model is in `.eval()` mode, the flag is
            automatically set to False always.
        :type optimize: bool, optional
        :param no_overlap: flag for performing optimization on the transpose
            continuous filter, defaults to False. The flag set to `True` should
            be used only when the filter positions do not overlap for different
            strides. RuntimeError will raise in case of non-compatible strides.
        :type no_overlap: bool, optional
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
            self._stride = Stride(stride)
        else:
            raise ValueError('stride must be dictionary.')

        self._net = model

        if isinstance(optimize, bool):
            self._optimize = optimize
        else:
            raise ValueError('optimize must be bool.')

        # choosing how to initialize based on optimization
        if self._optimize:
            # optimizing decorator ensure the function is called
            # just once
            self._choose_initialization = optimizing(
                self._initialize_convolution)
        else:
            self._choose_initialization = self._initialize_convolution

        if not isinstance(no_overlap, bool):
            raise ValueError('no_overlap must be bool.')

        if no_overlap:
            raise NotImplementedError
            self.transpose = self.transpose_no_overlap
        else:
            self.transpose = self.transpose_overlap

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
    def forward(self, X):
        pass

    @property
    @abstractmethod
    def transpose_overlap(self, X):
        pass

    @property
    @abstractmethod
    def transpose_no_overlap(self, X):
        pass

    @property
    @abstractmethod
    def _initialize_convolution(self, X, type):
        pass
