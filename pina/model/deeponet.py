"""Module for DeepONet model"""
import logging
from functools import partial, reduce

import torch
import torch.nn as nn

from pina import LabelTensor
from pina.model import FeedForward
from pina.utils import is_function


def check_combos(combos, variables):
    for combo in combos:
        for variable in combo:
            if variable not in variables:
                raise ValueError(
                    f"Combinations should be (overlapping) subsets of input variables, {variable} is not an input variable"
                )


def spawn_combo_networks(
    combos, layers, output_dimension, func, extra_feature, bias=True
):
    if not is_function(extra_feature):
        extra_feature_func = lambda _: extra_feature
    else:
        extra_feature_func = extra_feature

    return [
        FeedForward(
            layers=layers,
            input_variables=tuple(combo),
            output_variables=output_dimension,
            func=func,
            extra_features=extra_feature_func(combo),
            bias=bias,
        )
        for combo in combos
    ]


class DeepONet(torch.nn.Module):
    """
    The PINA implementation of DeepONet network.

    .. seealso::

        **Original reference**: Lu, L., Jin, P., Pang, G. et al. *Learning
        nonlinear operators via DeepONet based on the universal approximation
        theorem of operators*. Nat Mach Intell 3, 218–229 (2021).
        DOI: `10.1038/s42256-021-00302-5
        <https://doi.org/10.1038/s42256-021-00302-5>`_

    """

    def __init__(self, nets, output_variables, aggregator="*", reduction="+"):
        """
        :param iterable(torch.nn.Module) nets: Internal DeepONet networks
            (branch and trunk in the original DeepONet).
        :param list(str) output_variables: the list containing the labels
            corresponding to the components of the output computed by the
            model.
        :param string | callable aggregator: Aggregator to be used to aggregate
            partial results from the modules in `nets`. Partial results are
            aggregated component-wise. See :func:`_symbol_functions` for the
            available default aggregators.
        :param string | callable reduction: Reduction to be used to reduce
            the aggregated result of the modules in `nets` to the desired output
            dimension. See :func:`_symbol_functions` for the available default
            reductions.

        :Example:
            >>> branch = FFN(input_variables=['a', 'c'], output_variables=20)
            >>> trunk = FFN(input_variables=['b'], output_variables=20)
            >>> onet = DeepONet(nets=[trunk, branch], output_variables=output_vars)
            DeepONet(
              (trunk_net): FeedForward(
                (extra_features): Sequential()
                (model): Sequential(
                (0): Linear(in_features=1, out_features=20, bias=True)
                (1): Tanh()
                (2): Linear(in_features=20, out_features=20, bias=True)
                (3): Tanh()
                (4): Linear(in_features=20, out_features=20, bias=True)
                )
              )
              (branch_net): FeedForward(
                (extra_features): Sequential()
                (model): Sequential(
                (0): Linear(in_features=2, out_features=20, bias=True)
                (1): Tanh()
                (2): Linear(in_features=20, out_features=20, bias=True)
                (3): Tanh()
                (4): Linear(in_features=20, out_features=20, bias=True)
                )
              )
            )
        """
        super().__init__()

        self.output_variables = output_variables
        self.output_dimension = len(output_variables)

        self._init_aggregator(aggregator, n_nets=len(nets))
        hidden_size = nets[0].model[-1].out_features
        self._init_reduction(reduction, hidden_size=hidden_size)

        if not DeepONet._all_nets_same_output_layer_size(nets):
            raise ValueError("All networks should have the same output size")
        self._nets = torch.nn.ModuleList(nets)
        logging.info("Combo DeepONet children: %s", list(self.children()))

    @staticmethod
    def _symbol_functions(**kwargs):
        return {
            "+": partial(torch.sum, **kwargs),
            "*": partial(torch.prod, **kwargs),
            "mean": partial(torch.mean, **kwargs),
            "min": lambda x: torch.min(x, **kwargs).values,
            "max": lambda x: torch.max(x, **kwargs).values,
        }

    def _init_aggregator(self, aggregator, n_nets):
        aggregator_funcs = DeepONet._symbol_functions(dim=2)
        if aggregator in aggregator_funcs:
            aggregator_func = aggregator_funcs[aggregator]
        elif isinstance(aggregator, nn.Module) or is_function(aggregator):
            aggregator_func = aggregator
        elif aggregator == "linear":
            aggregator_func = nn.Linear(n_nets, len(self.output_variables))
        else:
            raise ValueError(f"Unsupported aggregation: {str(aggregator)}")

        self._aggregator = aggregator_func
        logging.info("Selected aggregator: %s", str(aggregator_func))

        # test the aggregator
        test = self._aggregator(torch.ones((20, 3, n_nets)))
        if test.ndim < 2 or tuple(test.shape)[:2] != (20, 3):
            raise ValueError(
                f"Invalid aggregator output shape: {(20, 3, n_nets)} -> {test.shape}"
            )

    def _init_reduction(self, reduction, hidden_size):
        reduction_funcs = DeepONet._symbol_functions(dim=2)
        if reduction in reduction_funcs:
            reduction_func = reduction_funcs[reduction]
        elif isinstance(reduction, nn.Module) or is_function(reduction):
            reduction_func = reduction
        elif reduction == "linear":
            reduction_func = nn.Linear(hidden_size, len(self.output_variables))
        else:
            raise ValueError("Unsupported reduction: %s", str(reduction))

        self._reduction = reduction_func
        logging.info("Selected reduction: %s", str(reduction))

        # test the reduction
        test = self._reduction(torch.ones((20, 3, hidden_size)))
        if test.ndim < 2 or tuple(test.shape)[:2] != (20, 3):
            raise ValueError(
                f"Invalid reduction output shape: {(20, 3, hidden_size)} -> {test.shape}"
            )

    @staticmethod
    def _all_nets_same_output_layer_size(nets):
        size = nets[0].layers[-1].out_features
        return all((net.layers[-1].out_features == size for net in nets[1:]))

    @property
    def input_variables(self):
        """The input variables of the model"""
        nets_input_variables = map(lambda net: net.input_variables, self._nets)
        return reduce(sum, nets_input_variables)

    def forward(self, x):
        """
        Defines the computation performed at every call.

        :param LabelTensor x: the input tensor.
        :return: the output computed by the model.
        :rtype: LabelTensor
        """

        nets_outputs = tuple(
            net(x.extract(net.input_variables)) for net in self._nets
        )
        # torch.dstack(nets_outputs): (batch_size, net_output_size, n_nets)
        aggregated = self._aggregator(torch.dstack(nets_outputs))
        # net_output_size = output_variables * hidden_size
        aggregated_reshaped = aggregated.view(
            (len(x), len(self.output_variables), -1)
        )
        output_ = self._reduction(aggregated_reshaped)
        output_ = torch.squeeze(torch.atleast_3d(output_), dim=2)

        assert output_.shape == (len(x), len(self.output_variables))

        output_ = output_.as_subclass(LabelTensor)
        output_.labels = self.output_variables
        return output_
