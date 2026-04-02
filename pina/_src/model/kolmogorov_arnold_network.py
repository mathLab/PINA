import torch
from pina._src.model.block.kan_block import KANBlock
from pina._src.core.utils import check_consistency


class KolmogorovArnoldNetwork(torch.nn.Module):
    """
    TODO: add docstring.
    """

    def __init__(
        self,
        layers,
        spline_order=3,
        n_knots=10,
        grid_range=[-1, 1],
        base_function=torch.nn.SiLU,
        use_base_linear=True,
        use_bias=True,
        init_scale_spline=1e-2,
        init_scale_base=1.0,
    ):
        """
        Initialization of the :class:`KolmogorovArnoldNetwork` class.

        :param layers: A list of integers specifying the sizes of each layer,
            including input and output dimensions.
        :type layers: list | tuple.
        :param int spline_order: The order of each spline basis function.
            Default is 3 (cubic splines).
        :param int n_knots: The number of knots for each spline basis function.
            Default is 3.
        :param grid_range: The range for the spline knots. It must be either a
            list or a tuple of the form [min, max]. Default is [0, 1].
        :type grid_range: list | tuple.
        :param torch.nn.Module base_function: The base activation function to be
            applied to the input before the linear transformation. Default is
            :class:`torch.nn.SiLU`.
        :param bool use_base_linear: Whether to include a linear transformation
            of the base function output. Default is True.
        :param bool use_bias: Whether to include a bias term in the output.
            Default is True.
        :param init_scale_spline: The scale for initializing each spline
            control points. Default is 1e-2.
        :type init_scale_spline: float | int.
        :param init_scale_base: The scale for initializing the base linear
            weights. Default is 1.0.
        :type init_scale_base: float | int.
        :raises ValueError: If ``grid_range`` is not of length 2.
        """
        super().__init__()

        # Check consistency -- all other checks are performed in KANBlock
        check_consistency(layers, int)
        if len(layers) < 2:
            raise ValueError(
                "`Provide at least two elements for layers (input and output)."
            )

        # Initialize KAN blocks
        self.kan_layers = torch.nn.ModuleList(
            [
                KANBlock(
                    input_dimensions=layers[i],
                    output_dimensions=layers[i + 1],
                    spline_order=spline_order,
                    n_knots=n_knots,
                    grid_range=grid_range,
                    base_function=base_function,
                    use_base_linear=use_base_linear,
                    use_bias=use_bias,
                    init_scale_spline=init_scale_spline,
                    init_scale_base=init_scale_base,
                )
                for i in range(len(layers) - 1)
            ]
        )

    def forward(self, x):
        """
        TODO: add docstring.
        """
        for layer in self.kan_layers:
            x = layer(x)

        return x
