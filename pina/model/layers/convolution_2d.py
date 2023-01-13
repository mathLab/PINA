"""Module for Continuous Convolution class."""

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
            >>> dim = [3, 3]
            >>> stride = {"domain": [10, 10],
                          "start": [0, 0],
                          "jumps": [3, 3],
                          "direction": [1, 1.]}
            >>> conv = ContinuousConv2D(1, 2, dim, stride)
            >>> conv
            ContinuousConv2D(
                (_net): ModuleList(
                    (0): NeuralNet(
                    (model): Sequential(
                        (0): Linear(in_features=2, out_features=20, bias=True)
                        (1): Tanh()
                        (2): Linear(in_features=20, out_features=20, bias=True)
                        (3): Tanh()
                        (4): Linear(in_features=20, out_features=1, bias=True)
                    )
                    )
                    (1): NeuralNet(
                    (model): Sequential(
                        (0): Linear(in_features=2, out_features=20, bias=True)
                        (1): Tanh()
                        (2): Linear(in_features=20, out_features=20, bias=True)
                        (3): Tanh()
                        (4): Linear(in_features=20, out_features=1, bias=True)
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

        if self._net is None:
            model = NeuralNet(len(self._dim), 1)

        self._net = torch.nn.ModuleList(model for _ in range(
            self._input_numb_field * self._output_numb_field))

        self._stride = create_stride(self._stride)

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

        for batch, x in enumerate(X):

            mapped_points = []
            indeces_channels = []

            for current_stride in self._stride:

                # indeces of points falling into filter range
                indeces = check_point(x, current_stride, self._dim)

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

            # for each output numb field
            idx_net = 0
            for out_idx in range(self._output_numb_field):
                # compute convolution
                res_tmp = []
                for idx in range(self._input_numb_field):
                    # extract input for each channel
                    single_channel_input = stacked_input[idx]
                    # extract filter
                    net = self._net[idx_net]
                    idx_net += 1
                    # caculate filter value
                    staked_output = net(single_channel_input[..., :-1])
                    # perform integral for all strides in one channel
                    integral = self._integral(staked_output,
                                              single_channel_input[..., -1],
                                              indeces_channels[0])
                    res_tmp.append(integral)

                # stacking integral results and summing over channels
                conv[batch, out_idx, :, -1] = torch.stack(res_tmp).sum(dim=0)

        return conv

    def transpose(self, integrals, grid):

        X = grid.clone().detach()
        conv_transposed = X.clone()

        for batch, x in enumerate(X):

            mapped_points = []
            indeces_channels = []

            for current_stride in self._stride:

                # indeces of points falling into filter range
                indeces = check_point(x, current_stride, self._dim)

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

            # for each output numb field
            idx_net = 0
            for _ in range(self._output_numb_field):
                # compute convolution
                res_tmp = []
                for idx in range(self._input_numb_field):
                    # extract input for each channel
                    single_channel_input = stacked_input[idx]
                    rep_idx = torch.tensor(indeces_channels[0])

                    integral = integrals[batch, idx,
                                         :].repeat_interleave(rep_idx)

                    # extract filter
                    net = self._net[idx_net]
                    idx_net += 1
                    # caculate filter value
                    staked_output = net(
                        single_channel_input[..., :-1]).flatten()
                    integral = staked_output * integral
                    res_tmp.append(integral)

                # stacking integral results and summing over channels
                conv_transposed[batch,
                                idx, :, -1] = torch.stack(res_tmp).sum(dim=0)

        return conv_transposed
