"""Module for DeepONet model"""

from ..utils import check_consistency, is_function
from .mionet import MIONet


class DeepONet(MIONet):
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

    def __init__(
        self,
        branch_net,
        trunk_net,
        input_indeces_branch_net,
        input_indeces_trunk_net,
        aggregator="*",
        reduction="+",
        scale=True,
        translation=True,
    ):
        """
        :param torch.nn.Module branch_net: The neural network to use as branch
            model. It has to take as input a
            :py:obj:`pina.label_tensor.LabelTensor` or :class:`torch.Tensor`.
            The number of dimensions of the output has to be the same of the
            ``trunk_net``.
        :param torch.nn.Module trunk_net: The neural network to use as trunk
            model. It has to take as input a
            :py:obj:`pina.label_tensor.LabelTensor` or :class:`torch.Tensor`.
            The number of dimensions of the output has to be the same of the
            ``branch_net``.
        :param list(int) or list(str) input_indeces_branch_net: List of indeces
            to extract from the input variable in the forward pass for the
            branch net. If a list of ``int`` is passed, the corresponding
            columns of the inner most entries are extracted. If a list of
            ``str`` is passed the variables of the corresponding
            :py:obj:`pina.label_tensor.LabelTensor` are extracted.
        :param list(int) or list(str) input_indeces_trunk_net: List of indeces
            to extract from the input variable in the forward pass for the trunk
            net. If a list of ``int`` is passed, the corresponding columns of
            the inner most entries are extracted. If a list of ``str`` is passed
            the variables of the corresponding
            :py:obj:`pina.label_tensor.LabelTensor` are extracted.
        :param str or Callable aggregator: Aggregator to be used to aggregate
            partial results from the modules in `nets`. Partial results are
            aggregated component-wise. Available aggregators include sum: ``+``,
            product: ``*``, mean: ``mean``, min: ``min``, max: ``max``.
        :param str or Callable reduction: Reduction to be used to reduce
            the aggregated result of the modules in `nets` to the desired output
            dimension. Available reductions include sum: ``+``, product: ``*``,
            mean: ``mean``, min: ``min``, max: ``max``.
        :param bool or Callable scale: Scaling the final output before returning
            the forward pass, default True.
        :param bool or Callable translation: Translating the final output before
            returning the forward pass, default True.

        .. warning::
            In the forward pass we do not check if the input is instance of
            :py:obj:`pina.label_tensor.LabelTensor` or :class:`torch.Tensor`. A
            general rule is that for a :py:obj:`pina.label_tensor.LabelTensor`
            input both list of integers and list of strings can be passed for
            ``input_indeces_branch_net`` and ``input_indeces_trunk_net``.
            Differently, for a :class:`torch.Tensor` only a list of integers can
            be passed for ``input_indeces_branch_net`` and
            ``input_indeces_trunk_net``.

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
        networks = {
            branch_net: input_indeces_branch_net,
            trunk_net: input_indeces_trunk_net,
        }
        super().__init__(
            networks=networks,
            aggregator=aggregator,
            reduction=reduction,
            scale=scale,
            translation=translation,
        )

    @property
    def branch_net(self):
        """
        The branch net for DeepONet.
        """
        return self.models[0]

    @property
    def trunk_net(self):
        """
        The trunk net for DeepONet.
        """
        return self.models[1]
