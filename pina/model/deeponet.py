"""Module for DeepONet model"""
import torch
import torch.nn as nn
from ..utils import check_consistency, is_function
from functools import partial


class DeepONet(torch.nn.Module):
    """
    The PINA implementation of DeepONet network.

    DeepONet is a general architecture for learning Operators. Unlike
    traditional machine learning methods DeepONet is designed to map
    entire functions to other functions. It can be trained both with 
    Physics Informed or Supervised learning strategies.

    .. seealso::

        **Original reference**: Lu, L., Jin, P., Pang, G. et al. *Learning
        nonlinear operators via DeepONet based on the universal approximation
        theorem of operators*. Nat Mach Intell 3, 218–229 (2021).
        DOI: `10.1038/s42256-021-00302-5
        <https://doi.org/10.1038/s42256-021-00302-5>`_

    """
    def __init__(self,
                 branch_net,
                 trunk_net,
                 input_indeces_branch_net,
                 input_indeces_trunk_net,
                 aggregator="*", 
                 reduction="+"):
        """
        :param torch.nn.Module branch_net: The neural network to use as branch
            model. It has to take as input a :class:`LabelTensor`
            or :class:`torch.Tensor`. The number of dimensions of the output has
            to be the same of the ``trunk_net``.
        :param torch.nn.Module trunk_net: The neural network to use as trunk
            model. It has to take as input a :class:`LabelTensor`
            or :class:`torch.Tensor`. The number of dimensions of the output
            has to be the same of the ``branch_net``.
        :param list(int) | list(str) input_indeces_branch_net: List of indeces
            to extract from the input variable in the forward pass for the
            branch net. If a list of ``int`` is passed, the corresponding columns
            of the inner most entries are extracted. If a list of ``str`` is passed
            the variables of the corresponding :class:`LabelTensor` are extracted.
        :param list(int) | list(str) input_indeces_trunk_net: List of indeces
            to extract from the input variable in the forward pass for the
            trunk net. If a list of ``int`` is passed, the corresponding columns
            of the inner most entries are extracted. If a list of ``str`` is passed
            the variables of the corresponding :class:`LabelTensor` are extracted.
        :param str | callable aggregator: Aggregator to be used to aggregate
            partial results from the modules in `nets`. Partial results are
            aggregated component-wise. See
            :func:`pina.model.deeponet.DeepONet._symbol_functions` for the
            available default aggregators.
        :param str | callable reduction: Reduction to be used to reduce
            the aggregated result of the modules in `nets` to the desired output
            dimension. See :py:obj:`pina.model.deeponet.DeepONet._symbol_functions` for the available default
            reductions. 

        .. warning::
            In the forward pass we do not check if the input is instance of
            :class:`LabelTensor` or :class:`torch.Tensor`. A general rule is
            that for a :class:`LabelTensor` input both list of integers and
            list of strings can be passed for ``input_indeces_branch_net``
            and ``input_indeces_trunk_net``. Differently, for a :class:`torch.Tensor`
            only a list of integers can be passed for ``input_indeces_branch_net``
            and ``input_indeces_trunk_net``.

        :Example:
            >>> branch_net = FeedForward(input_dimensons=1, output_dimensions=10)
            >>> trunk_net = FeedForward(input_dimensons=1, output_dimensions=10)
            >>> model = DeepONet(branch_net=branch_net,
            ...                  trunk_net=trunk_net,
            ...                  input_indeces_branch_net=['x'],
            ...                  input_indeces_trunk_net=['t'],
            ...                  reduction='+',
            ...                  aggregator='*')
            >>> model
            DeepONet(
            (trunk_net): FeedForward(
                (model): Sequential(
                (0): Linear(in_features=1, out_features=20, bias=True)
                (1): Tanh()
                (2): Linear(in_features=20, out_features=20, bias=True)
                (3): Tanh()
                (4): Linear(in_features=20, out_features=10, bias=True)
                )
            )
            (branch_net): FeedForward(
                (model): Sequential(
                (0): Linear(in_features=1, out_features=20, bias=True)
                (1): Tanh()
                (2): Linear(in_features=20, out_features=20, bias=True)
                (3): Tanh()
                (4): Linear(in_features=20, out_features=10, bias=True)
                )
            )
            )
        """
        super().__init__()

        # check type consistency
        check_consistency(input_indeces_branch_net, (str, int))
        check_consistency(input_indeces_trunk_net, (str, int))
        check_consistency(branch_net, torch.nn.Module)
        check_consistency(trunk_net, torch.nn.Module)

        # check trunk branch nets consistency
        trunk_out_dim = trunk_net.layers[-1].out_features
        branch_out_dim = branch_net.layers[-1].out_features
        if trunk_out_dim != branch_out_dim:
            raise ValueError('Branch and trunk networks have not the same '
                             'output dimension.')

        # assign trunk and branch net with their input indeces
        self.trunk_net = trunk_net
        self._trunk_indeces = input_indeces_trunk_net
        self.branch_net = branch_net
        self._branch_indeces = input_indeces_branch_net

        # initializie aggregation
        self._init_aggregator(aggregator=aggregator)
        self._init_reduction(reduction=reduction)

        # scale and translation
        self._scale = torch.nn.Parameter(torch.tensor([1.0]))
        self._trasl = torch.nn.Parameter(torch.tensor([0.0]))

    @staticmethod
    def _symbol_functions(**kwargs):
        """
        Return a dictionary of functions that can be used as aggregators or
        reductions.
        """
        return {
            "+": partial(torch.sum, **kwargs),
            "*": partial(torch.prod, **kwargs),
            "mean": partial(torch.mean, **kwargs),
            "min": lambda x: torch.min(x, **kwargs).values,
            "max": lambda x: torch.max(x, **kwargs).values,
        }
    
    def _init_aggregator(self, aggregator):
        aggregator_funcs = DeepONet._symbol_functions(dim=2)
        if aggregator in aggregator_funcs:
            aggregator_func = aggregator_funcs[aggregator]
        elif isinstance(aggregator, nn.Module) or is_function(aggregator):
            aggregator_func = aggregator
        else:
            raise ValueError(f"Unsupported aggregation: {str(aggregator)}")

        self._aggregator = aggregator_func


    def _init_reduction(self, reduction):
        reduction_funcs = DeepONet._symbol_functions(dim=-1)
        if reduction in reduction_funcs:
            reduction_func = reduction_funcs[reduction]
        elif isinstance(reduction, nn.Module) or is_function(reduction):
            reduction_func = reduction
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

        self._reduction = reduction_func

    def _get_vars(self, x, indeces):
        if isinstance(indeces[0], str):
            try:
                return x.extract(indeces)
            except AttributeError:
                raise RuntimeError('Not possible to extract input variables from tensor.'
                                   ' Ensure that the passed tensor is a LabelTensor or'
                                   ' pass list of integers to extract variables. For'
                                   ' more information refer to warning in the documentation.')
        elif isinstance(indeces[0], int):
            return x[..., indeces]
        else:
            raise RuntimeError('Not able to extract right indeces for tensor.'
                               ' For more information refer to warning in the documentation.')
        
    def forward(self, x):
        """
        Defines the computation performed at every call.

        :param LabelTensor | torch.Tensor x: The input tensor for the forward call.
        :return: The output computed by the DeepONet model.
        :rtype: LabelTensor | torch.Tensor 
        """

        # forward pass
        branch_output = self.branch_net(self._get_vars(x, self._branch_indeces))
        trunk_output = self.trunk_net(self._get_vars(x, self._trunk_indeces))

        # aggregation
        aggregated = self._aggregator(torch.dstack((branch_output, trunk_output)))

        # reduce
        output_ = self._reduction(aggregated).reshape(-1, 1)

        # scale and translate 
        output_ *= self._scale
        output_ += self._trasl

        return output_