"""Module Averaging Neural Operator."""

import torch
from torch import nn, cat
from .block import AVNOBlock
from .kernel_neural_operator import KernelNeuralOperator
from ..utils import check_consistency


class AveragingNeuralOperator(KernelNeuralOperator):
    """
    Implementation of Averaging Neural Operator.

    Averaging Neural Operator is a general architecture for
    learning Operators. Unlike traditional machine learning methods
    AveragingNeuralOperator is designed to map entire functions
    to other functions. It can be trained with Supervised learning strategies.
    AveragingNeuralOperator does convolution by performing a field average.

    .. seealso::

        **Original reference**: Lanthaler S. Li, Z., Kovachki,
        Stuart, A. (2020). *The Nonlocal Neural Operator:
        Universal Approximation*.
        DOI: `arXiv preprint arXiv:2304.13221.
        <https://arxiv.org/abs/2304.13221>`_
    """

    def __init__(
        self,
        lifting_net,
        projecting_net,
        field_indices,
        coordinates_indices,
        n_layers=4,
        func=nn.GELU,
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
        :param int n_layers: number of hidden layers. Default is 4.
        :param torch.nn.Module func: the activation function to use,
            default to torch.nn.GELU.
        """

        # check consistency
        check_consistency(field_indices, str)
        check_consistency(coordinates_indices, str)
        check_consistency(n_layers, int)
        check_consistency(func, nn.Module, subclass=True)

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
                "The projecting_net input must be equal to"
                "the embedding dimension (which is the output) "
                "of the lifting_net plus the dimension of the "
                "coordinates, i.e. len(coordinates_indices)."
            )

        # assign
        self.coordinates_indices = coordinates_indices
        self.field_indices = field_indices
        integral_net = nn.Sequential(
            *[AVNOBlock(output_lifting_net, func) for _ in range(n_layers)]
        )
        super().__init__(lifting_net, integral_net, projecting_net)

    def forward(self, x):
        r"""
        Forward computation for Averaging Neural Operator. It performs a
        lifting of the input by the ``lifting_net``. Then different layers
        of Averaging Neural Operator Blocks are applied.
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
        points_tmp = x.extract(self.coordinates_indices)
        new_batch = x.extract(self.field_indices)
        new_batch = cat((new_batch, points_tmp), dim=-1)
        new_batch = self._lifting_operator(new_batch)
        new_batch = self._integral_kernels(new_batch)
        new_batch = cat((new_batch, points_tmp), dim=-1)
        new_batch = self._projection_operator(new_batch)
        return new_batch
