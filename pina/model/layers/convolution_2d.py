"""Module for Continuous Convolution class."""

import copy
from convolution import BaseContinuousConv
from utils import Integral, NeuralNet, create_stride, check_point, map_points_
import torch


class ContinuousConv2D(BaseContinuousConv):
    """
    Implementation of Continuous Convolutional operator.

    .. seealso::

    **Original reference**: Coscia, D., Meneghetti, L., Demo, N.,
    Stabile, G., & Rozza, G.. (2022). A Continuous Convolutional Trainable
    Filter for Modelling Unstructured Data.
    DOI: `10.48550/arXiv.2210.13416
    <https://doi.org/10.48550/arXiv.2210.13416>`_

    """

    def __init__(self, input_numb_field, output_numb_field,
                 filter_dim, stride, model=None):
        """Continuous Convolution 2D implementation

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



        :Example:
            >>> class MLP(torch.nn.Module):
                    def __init__(self) -> None:
                        super().__init__()
                        self. model = torch.nn.Sequential(torch.nn.Linear(2, 8),
                                                        torch.nn.ReLU(),
                                                        torch.nn.Linear(8, 8),
                                                        torch.nn.ReLU(),
                                                        torch.nn.Linear(8, 1))

                    def forward(self, x):
                        return self.model(x)
            >>> dim = [3, 3]
            >>> stride = {"domain": [10, 10],
                          "start": [0, 0],
                          "jumps": [3, 3],
                          "direction": [1, 1.]}
            >>> conv = ContinuousConv2D(1, 2, dim, stride, MLP)
            >>> conv
                ContinuousConv2D(
                (_net): ModuleList(
                    (0): MLP(
                    (model): Sequential(
                        (0): Linear(in_features=2, out_features=8, bias=True)
                        (1): ReLU()
                        (2): Linear(in_features=8, out_features=8, bias=True)
                        (3): ReLU()
                        (4): Linear(in_features=8, out_features=1, bias=True)
                    )
                    )
                    (1): MLP(
                    (model): Sequential(
                        (0): Linear(in_features=2, out_features=8, bias=True)
                        (1): ReLU()
                        (2): Linear(in_features=8, out_features=8, bias=True)
                        (3): ReLU()
                        (4): Linear(in_features=8, out_features=1, bias=True)
                    )
                    )
                )
                )
        """

        super().__init__(input_numb_field=input_numb_field,
                         output_numb_field=output_numb_field,
                         filter_dim=filter_dim,
                         stride=stride,
                         model=model)

        self._integral = Integral('discrete')

        # create the network
        self._net = self._spawn_networks(model)

        # create the stride list of points
        self._stride = create_stride(self._stride)

    def _spawn_networks(self, model):
        nets = []
        if self._net is None:
            for _ in range(self._input_numb_field * self._output_numb_field):
                tmp = NeuralNet(len(self._dim), 1)
                nets.append(tmp)
        else:
            if not isinstance(model, object):
                raise ValueError("Expected a python class inheriting"
                                 " from torch.nn.Module")

            for _ in range(self._input_numb_field * self._output_numb_field):
                tmp = model()
                if not isinstance(tmp, torch.nn.Module):
                    raise ValueError("The python class must be inherited from"
                                     " torch.nn.Module. See the docstring for"
                                     " an example.")
                nets.append(tmp)

        return torch.nn.ModuleList(nets)

    def _make_grid(self, batch_dim):
        # filter dimension + number of points in output grid
        filter_dim = len(self._dim)
        number_points = len(self._stride)

        # initialize the grid
        grid = torch.zeros(size=(batch_dim,
                                 self._output_numb_field,
                                 number_points,
                                 filter_dim + 1))
        grid[..., :-1] = (self._stride / self._dim)

        return grid.detach()

    def _extract_mapped_points(self, batch_idx, index, x):
        """Priviate method to extract mapped points in the filter

        :param x: input tensor [channel x N x dim]
        :type x: torch.tensor
        :return: mapped points and indeces for each channel
        :rtype: tuple(torch.tensor, list)
        """
        mapped_points = []
        indeces_channels = []

        for stride_idx, current_stride in enumerate(self._stride):
            # indeces of points falling into filter range
            indeces = index[stride_idx][batch_idx]

            # how many points for each channel fall into the filter?
            numb_points_insiede = torch.sum(indeces, dim=-1).tolist()

            # extracting points for each channel
            # shape: [sum(numb_points_insiede), filter_dim + 1]
            point_stride = x[indeces]

            # mapping points in filter domain
            map_points_(point_stride[..., :-1], current_stride)

            # extracting points for each channel
            point_stride_channel = point_stride.split(numb_points_insiede)

            # appending in list for later use
            mapped_points.append(point_stride_channel)
            indeces_channels.append(numb_points_insiede)

        # stacking input for passing to neural net
        mapping = map(torch.cat, zip(*mapped_points))
        stacked_input = tuple(mapping)
        indeces_channels = tuple(zip(*indeces_channels))

        return stacked_input, indeces_channels

    def _find_index(self, X):
        """Private method to extract indeces for convolution

        :param X: input tensor, as in ContinuousConv2D docstring
        :type X: torch.tensor
        """
        index = []
        for _, current_stride in enumerate(self._stride):

            tmp = check_point(X, current_stride, self._dim)
            index.append(tmp)

        self._index = index

    def forward(self, X):
        """Forward pass in the FFConv layer

        :param x: input data ( input_numb_field x N x filter_dim )
        :type x: torch.tensor
        :return: feed forward convolution ( output_numb_field x N x filter_dim )
        :rtype: torch.tensor
        """
        # batch dimension
        batch_dim = X.shape[0]
        conv = self._make_grid(batch_dim)

        # extracting indeces for convolution on batches
        # TODO: if we want to optimize and we know that
        # the grid is the same for all samples in the
        # training set, we can do something like call this
        # function just once when we are in .train() mode
        self._find_index(X)

        for batch_idx, x in enumerate(X):

            # extract mapped points
            stacked_input, indeces_channels = self._extract_mapped_points(
                batch_idx, self._index, x)

            # for each output numb field
            tot_dim = self._output_numb_field * self._input_numb_field
            res_tmp = []
            for idx_conv in range(tot_dim):
                # compute convolution
                idx = idx_conv % self._input_numb_field
                # extract input for each channel
                single_channel_input = stacked_input[idx]
                # extract filter
                net = self._net[idx_conv]
                # caculate filter value
                staked_output = net(single_channel_input[..., :-1])
                # perform integral for all strides in one channel
                integral = self._integral(staked_output,
                                          single_channel_input[..., -1],
                                          indeces_channels[idx])
                res_tmp.append(integral)

            # stacking integral results and summing over channels
            res_tmp = torch.stack(res_tmp)
            conv[batch_idx, ..., -1] = res_tmp.reshape(self._output_numb_field,
                                                       self._input_numb_field,
                                                       -1).sum(1)

        return conv

    def transpose(self, integrals, grid):

        X = grid.clone().detach()
        conv_transposed = X.clone()

        # extracting indeces for convolution on batches
        # TODO: if we want to optimize and we know that
        # the grid is the same for all samples in the
        # training set, we can do something like call this
        # function just once when we are in .train() mode
        self._find_index(X)

        for batch_idx, x in enumerate(X):

            # extract mapped points
            stacked_input, indeces_channels = self._extract_mapped_points(
                batch_idx, self._index, x)

            # for each output numb field
            res_tmp = []
            tot_dim = self._output_numb_field * self._input_numb_field
            for idx_conv in range(tot_dim):
                # compute convolution
                idx = idx_conv % self._input_numb_field
                # extract input for each channel
                single_channel_input = stacked_input[idx]
                rep_idx = torch.tensor(indeces_channels[idx])
                integral = integrals[batch_idx,
                                     idx, :].repeat_interleave(rep_idx)
                # extract filter
                net = self._net[idx_conv]

                # caculate filter value
                staked_output = net(single_channel_input[..., :-1]).flatten()
                integral = staked_output * integral
                res_tmp.append(integral)

            # stacking integral results and summing over channels
            res_tmp = torch.stack(res_tmp).reshape(self._input_numb_field,
                                                   self._output_numb_field,
                                                   -1).sum(1)
            conv_transposed[batch_idx, ..., -1] = res_tmp

        return conv_transposed
