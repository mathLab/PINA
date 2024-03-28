"""Module LowRank Neural Operator."""

import torch
from torch import nn, concatenate
from .base_no import KernelNeuralOperator
from pina.utils import check_consistency
from .layers.lowrank_layer import LowRankBlock


class LowRankNeuralOperator(KernelNeuralOperator):
    """
    Implementation of LowRank Neural Operator.

    LowRank Neural Operator is a general architecture for
    learning Operators. Unlike traditional machine learning methods
    LowRankNeuralOperator is designed to map entire functions
    to other functions. It can be trained with Supervised or PINN based
    learning strategies.
    LowRankNeuralOperator does convolution by performing a low rank
    approximation, see :class:`~pina.model.layers.lowrank_layer.LowRankBlock`.

    .. seealso::

        **Original reference**: Kovachki, N., Li, Z., Liu, B.,
        Azizzadenesheli, K., Bhattacharya, K., Stuart, A., & Anandkumar, A.
        (2023). *Neural operator: Learning maps between function
        spaces with applications to PDEs*. Journal of Machine Learning
        Research, 24(89), 1-97.
    """

    def __init__(
        self,
        lifting_net,
        projecting_net,
        field_indices,
        coordinates_indices,
        n_kernel_layers,
        rank,
        inner_size=20,
        n_layers=2,
        func=torch.nn.Tanh,
        bias=True
    ):
        """
        :param torch.nn.Module lifting_net: The neural network for lifting
            the input. It must take as input the input field and the coordinates
            at which the input field is avaluated. The output of the lifting
            net is chosen as embedding dimension of the problem
        :param torch.nn.Module projecting_net: The neural network for
            projecting the output. It must take as input the embedding dimension
            (output of the ``lifting_net``) plus the dimension
            of the coordinates.
        :param list[str] field_indices: the label of the fields
            in the input tensor.
        :param list[str] coordinates_indices: the label of the
            coordinates in the input tensor.
        :param int n_kernel_layers: number of hidden kernel layers.
            Default is 4.
        :param int inner_size: Number of neurons in the hidden layer(s) for the
            basis function network. Default is 20.
        :param int n_layers: Number of hidden layers. for the
            basis function network. Default is 2.
        :param func: The activation function to use for the
            basis function network. If a single
            :class:`torch.nn.Module` is passed, this is used as
            activation function after any layers, except the last one.
            If a list of Modules is passed,
            they are used as activation functions at any layers, in order.
        :param bool bias: If ``True`` the MLP will consider some bias for the
            basis function network.
        """

        # check consistency
        check_consistency(field_indices, str)
        check_consistency(coordinates_indices, str)
        check_consistency(n_kernel_layers, int)

        # check hidden dimensions match
        input_lifting_net = next(lifting_net.parameters()).size()[-1]
        output_lifting_net = lifting_net(
            torch.rand(size=next(lifting_net.parameters()).size())
        ).shape[-1]
        projecting_net_input = next(projecting_net.parameters()).size()[-1]

        if len(field_indices) + len(coordinates_indices) != input_lifting_net:
            raise ValueError(
                "The lifting_net must take as input the "
                "coordinates vector and the field vector."
            )

        if (
            output_lifting_net + len(coordinates_indices)
            != projecting_net_input
        ):
            raise ValueError(
                "The projecting_net input must be equal to "
                "the embedding dimension (which is the output) "
                "of the lifting_net plus the dimension of the "
                "coordinates, i.e. len(coordinates_indices)."
            )

        # assign
        self.coordinates_indices = coordinates_indices
        self.field_indices = field_indices
        integral_net = nn.Sequential(
            *[LowRankBlock(input_dimensions=len(coordinates_indices),
                           embedding_dimenion=output_lifting_net,
                           rank=rank,
                           inner_size=inner_size,
                           n_layers=n_layers,
                           func=func,
                           bias=bias) for _ in range(n_kernel_layers)]
        )
        super().__init__(lifting_net, integral_net, projecting_net)

    def forward(self, x):
        r"""
        Forward computation for LowRank Neural Operator. It performs a
        lifting of the input by the ``lifting_net``. Then different layers
        of LowRank Neural Operator Blocks are applied.
        Finally the output is projected to the final dimensionality
        by the ``projecting_net``.

        :param torch.Tensor x: The input tensor for fourier block,
            depending on ``dimension`` in the initialization. It expects
            a tensor :math:`B \times N \times D`,
            where :math:`B` is the batch_size, :math:`N` the number of points
            in the mesh, :math:`D` the dimension of the problem, i.e. the sum
            of ``len(coordinates_indices)+len(field_indices)``.
        :return: The output tensor obtained from Average Neural Operator.
        :rtype: torch.Tensor
        """
        # extract points
        coords = x.extract(self.coordinates_indices)
        # lifting
        x = self._lifting_operator(x)
        # kernel
        for module in self._integral_kernels:
            x = module(x, coords)
        # projecting
        return self._projection_operator(concatenate((x, coords), dim=-1))
