"""Module for Continuous Convolution class."""

from .convolution import BaseContinuousConv
from .utils_convolution import check_point, map_points_
from .integral import Integral
from ..feed_forward import FeedForward
import torch


class ContinuousConv(BaseContinuousConv):
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
                 filter_dim, stride, model=None, optimize=False,
                 no_overlap=False):
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
        of the problem (coordinates + field value).

        :param input_numb_field: Number of fields $N_{in}$ in the input.
        :type input_numb_field: int
        :param output_numb_field: Number of fields $N_{out}$  in the output.
        :type output_numb_field: int
        :param filter_dim: Dimension of the filter.
        :type filter_dim: tuple/ list
        :param stride: Stride for the filter.
        :type stride: dict
        :param model: Neural network for inner parametrization,
            defaults to None. If None, pina.FeedForward is used, more
            on https://mathlab.github.io/PINA/_rst/fnn.html.
        :type model: torch.nn.Module, optional
        :param optimize: Flag for performing optimization on the continuous
            filter, defaults to False. The flag `optimize=True` should be
            used only when the scatter datapoints are fixed through the
            training. If torch model is in `.eval()` mode, the flag is
            automatically set to False always.
        :type optimize: bool, optional
        :param no_overlap: Flag for performing optimization on the transpose
            continuous filter, defaults to False. The flag set to `True` should
            be used only when the filter positions do not overlap for different
            strides. RuntimeError will raise in case of non-compatible strides.
        :type no_overlap: bool, optional

        :Note: Using `optimize=True` the filter can be use either in `forward` or
            in `transpose` mode, not both. If `optimize=False` the same filter can
            be used for both `transpose` and `forward` modes.

        :Example:
            >>> class MLP(torch.nn.Module):
                    def __init__(self) -> None:
                        super().__init__()
                        self. model = torch.nn.Sequential(
                                                        torch.nn.Linear(2, 8),
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
                         model=model,
                         optimize=optimize,
                         no_overlap=no_overlap)

        # integral routine
        self._integral = Integral('discrete')

        # create the network
        self._net = self._spawn_networks(model)

        # stride for continuous convolution overridden
        self._stride = self._stride._stride_discrete

    def _spawn_networks(self, model):
        """Private method to create a collection of kernels

        :param model: a torch.nn.Module model in form of Object class
        :type model: torch.nn.Module
        :return: list of torch.nn.Module models
        :rtype: torch.nn.ModuleList

        """
        nets = []
        if self._net is None:
            for _ in range(self._input_numb_field * self._output_numb_field):
                tmp = FeedForward(len(self._dim), 1)
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
        """Private method to extract indeces for convolution.

        :param X: input tensor, as in ContinuousConv2D docstring
        :type X: torch.tensor

        """
        # append the index for each stride
        index = []
        for _, current_stride in enumerate(self._stride):

            tmp = check_point(X, current_stride, self._dim)
            index.append(tmp)

        # storing the index
        self._index = index

    def _make_grid_forward(self, X):
        """Private method to create forward convolution grid.

        :param X: input tensor, as in ContinuousConv2D docstring
        :type X: torch.tensor

        """
        # filter dimension + number of points in output grid
        filter_dim = len(self._dim)
        number_points = len(self._stride)

        # initialize the grid
        grid = torch.zeros(size=(X.shape[0],
                                 self._output_numb_field,
                                 number_points,
                                 filter_dim + 1),
                           device=X.device,
                           dtype=X.dtype)
        grid[..., :-1] = (self._stride + self._dim * 0.5)

        # saving the grid
        self._grid = grid.detach()

    def _make_grid_transpose(self, X):
        """Private method to create transpose convolution grid.

        :param X: input tensor, as in ContinuousConv2D docstring
        :type X: torch.tensor

        """
        # initialize to all zeros
        tmp = torch.zeros_like(X)
        tmp[..., :-1] = X[..., :-1]

        # save on tmp
        self._grid_transpose = tmp

    def _make_grid(self, X, type):
        """Private method to create convolution grid.

        :param X: input tensor, as in ContinuousConv2D docstring
        :type X: torch.tensor
        :param type: type of convolution, ['forward', 'inverse'] the
            possibilities
        :type type: string

        """
        # choose the type of convolution
        if type == 'forward':
            return self._make_grid_forward(X)
        elif type == 'inverse':
            self._make_grid_transpose(X)
        else:
            raise TypeError

    def _initialize_convolution(self, X, type='forward'):
        """Private method to intialize the convolution.
        The convolution is initialized by setting a grid and
        calculate the index for finding the points inside the
        filter.

        :param X: input tensor, as in ContinuousConv2D docstring
        :type X: torch.tensor
        :param type: type of convolution, ['forward', 'inverse'] the
            possibilities
        :type type: string
        """

        # variable for the convolution
        self._make_grid(X, type)

        # calculate the index
        self._find_index(X)

    def forward(self, X):
        """Forward pass in the Continuous Convolutional layer

        :param x: input data (input_numb_field x N x filter_dim)
        :type x: torch.tensor
        :return: feed forward convolution (output_numb_field x N x filter_dim)
        :rtype: torch.tensor
        """

        # initialize convolution
        if self.training:  # we choose what to do based on optimization
            self._choose_initialization(X, type='forward')

        else:  # we always initialize on testing
            self._initialize_convolution(X, 'forward')

        # create convolutional array
        conv = self._grid.clone().detach()

        # total number of fields
        tot_dim = self._output_numb_field * self._input_numb_field

        for batch_idx, x in enumerate(X):

            # extract mapped points
            stacked_input, indeces_channels = self._extract_mapped_points(
                batch_idx, self._index, x)

            # compute the convolution

            # storing intermidiate results for each channel convolution
            res_tmp = []
            # for each field
            for idx_conv in range(tot_dim):
                # index for each input field
                idx = idx_conv % self._input_numb_field
                # extract input for each channel
                single_channel_input = stacked_input[idx]
                # extract filter
                net = self._net[idx_conv]
                # calculate filter value
                staked_output = net(single_channel_input[..., :-1])
                # perform integral for all strides in one field
                integral = self._integral(staked_output,
                                          single_channel_input[..., -1],
                                          indeces_channels[idx])
                res_tmp.append(integral)

            # stacking integral results
            res_tmp = torch.stack(res_tmp)

            # sum filters (for each input fields) in groups
            # for different ouput fields
            conv[batch_idx, ..., -1] = res_tmp.reshape(self._output_numb_field,
                                                       self._input_numb_field,
                                                       -1).sum(1)
        return conv

    def transpose_no_overlap(self, integrals, X):
        """Transpose pass in the Continuous Convolutional layer for
            no-overlapping filters.

        :param integrals: Weights for the transpose convolution. Shape
            $$[B \times N_{in} \times N]$$
            where $B$ is the batch_size, $N_{in}$ is the number of input
            fields, $N$ the number of points in the mesh, $D$ the dimension
            of the problem.
        :type integral: torch.tensor
        :param X: Input data. Expect tensor of shape
            $$[B \times N_{in} \times M \time D]$$ where $B$ is the batch_size,
            $N_{in}$ is the number of input fields, $M$ the number of points
            in the mesh, $D$ the dimension of the problem. Note, last column
        :type X: torch.tensor
        :return: Feed forward transpose convolution. Tensor of shape
            $$[B \times N_{out} \times N]$$ where $B$ is the batch_size,
            $N_{out}$ is the number of output fields, $N$ the number of points
            in the mesh, $D$ the dimension of the problem.
        :rtype: torch.tensor

        Note: This function is automatically called when `.transpose()`
            method is used and `no_overlap=False`
        """

        # initialize convolution
        if self.training:  # we choose what to do based on optimization
            self._choose_initialization(X, type='inverse')

        else:  # we always initialize on testing
            self._initialize_convolution(X, 'inverse')

        # initialize grid
        X = self._grid_transpose.clone().detach()
        conv_transposed = self._grid_transpose.clone().detach()

        # total number of dim
        tot_dim = self._input_numb_field * self._output_numb_field

        for batch_idx, x in enumerate(X):

            # extract mapped points
            stacked_input, indeces_channels = self._extract_mapped_points(
                batch_idx, self._index, x)

            # compute the transpose convolution

            # total number of fields
            res_tmp = []

            # for each field
            for idx_conv in range(tot_dim):
                # index for each output field
                idx = idx_conv % self._output_numb_field
                # index for each input field
                idx_in = idx_conv % self._input_numb_field
                # extract input for each field
                single_channel_input = stacked_input[idx]
                rep_idx = torch.tensor(indeces_channels[idx])
                integral = integrals[batch_idx,
                                     idx_in, :].repeat_interleave(rep_idx)
                # extract filter
                net = self._net[idx_conv]
                # perform transpose convolution for all strides in one field
                staked_output = net(single_channel_input[..., :-1]).flatten()
                integral = staked_output * integral
                res_tmp.append(integral)

            # stacking integral results and sum
            # filters (for each input fields) in groups
            # for different output fields
            res_tmp = torch.stack(res_tmp).reshape(self._input_numb_field,
                                                   self._output_numb_field,
                                                   -1).sum(0)
            conv_transposed[batch_idx, ..., -1] = res_tmp

        return conv_transposed

    def transpose_overlap(self, integrals, X):
        """Transpose pass in the Continuous Convolutional layer for
            overlapping filters.

        :param integrals: Weights for the transpose convolution. Shape
            $$[B \times N_{in} \times N]$$
            where $B$ is the batch_size, $N_{in}$ is the number of input
            fields, $N$ the number of points in the mesh, $D$ the dimension
            of the problem.
        :type integral: torch.tensor
        :param X: Input data. Expect tensor of shape
            $$[B \times N_{in} \times M \time D]$$ where $B$ is the batch_size,
            $N_{in}$ is the number of input fields, $M$ the number of points
            in the mesh, $D$ the dimension of the problem. Note, last column
        :type X: torch.tensor
        :return: Feed forward transpose convolution. Tensor of shape
            $$[B \times N_{out} \times N]$$ where $B$ is the batch_size,
            $N_{out}$ is the number of output fields, $N$ the number of points
            in the mesh, $D$ the dimension of the problem.
        :rtype: torch.tensor

        Note: This function is automatically called when `.transpose()`
            method is used and `no_overlap=False`
        """

        # initialize convolution
        if self.training:  # we choose what to do based on optimization
            self._choose_initialization(X, type='inverse')

        else:  # we always initialize on testing
            self._initialize_convolution(X, 'inverse')

        # initialize grid
        X = self._grid_transpose.clone().detach()
        conv_transposed = self._grid_transpose.clone().detach()

        # list to iterate for calculating nn output
        tmp = [i for i in range(self._output_numb_field)]
        iterate_conv = [item for item in tmp for _ in range(
            self._input_numb_field)]

        for batch_idx, x in enumerate(X):

            # accumulator for the convolution on different batches
            accumulator_batch = torch.zeros(
                size=(self._grid_transpose.shape[1],
                      self._grid_transpose.shape[2]),
                requires_grad=True,
                device=X.device,
                dtype=X.dtype).clone()

            for stride_idx, current_stride in enumerate(self._stride):
                # indeces of points falling into filter range
                indeces = self._index[stride_idx][batch_idx]

                # number of points for each channel
                numb_pts_channel = tuple(indeces.sum(dim=-1))

                # extracting points for each channel
                point_stride = x[indeces]

                # if no points to upsample we just skip
                if point_stride.nelement() == 0:
                    continue

                # mapping points in filter domain
                map_points_(point_stride[..., :-1], current_stride)

                # input points for kernels
                # we split for extracting number of points for each channel
                nn_input_pts = point_stride[..., :-1].split(numb_pts_channel)

                # accumulate partial convolution results for each field
                res_tmp = []

                # for each channel field compute transpose convolution
                for idx_conv, idx_channel_out in enumerate(iterate_conv):

                    # index for input channels
                    idx_channel_in = idx_conv % self._input_numb_field

                    # extract filter
                    net = self._net[idx_conv]

                    # calculate filter value
                    staked_output = net(nn_input_pts[idx_channel_out])

                    # perform integral for all strides in one field
                    integral = staked_output * integrals[batch_idx,
                                                         idx_channel_in,
                                                         stride_idx]
                    # append results
                    res_tmp.append(integral.flatten())

                # computing channel sum
                channel_sum = []
                start = 0
                for _ in range(self._output_numb_field):
                    tmp = res_tmp[start:start + self._input_numb_field]
                    tmp = torch.vstack(tmp).sum(dim=0)
                    channel_sum.append(tmp)
                    start += self._input_numb_field

                # accumulate the results
                accumulator_batch[indeces] += torch.hstack(channel_sum)

            # save results of accumulation for each batch
            conv_transposed[batch_idx, ..., -1] = accumulator_batch

        return conv_transposed
