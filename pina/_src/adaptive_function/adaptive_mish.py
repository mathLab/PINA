"""Module for the Adaptive Mish activation function."""

import torch
from pina._src.adaptive_function.base_adaptive_function import (
    BaseAdaptiveFunction,
)


class AdaptiveMish(BaseAdaptiveFunction):
    r"""
    Adaptive, trainable variant of the :class:`~torch.nn.Mish` activation.

    This module extends the standard Mish by introducing learnable scaling
    and shifting parameters applied to both the input and the output.

    Given the function :math:`\text{Mish}:\mathbb{R}^n\rightarrow\mathbb{R}^n`,
    the corresponding adaptive activation
    :math:`\text{Mish}_{\text{adaptive}}:\mathbb{R}^n\rightarrow\mathbb{R}^n` is
    defined as:

    .. math::
        \text{Mish}_{\text{adaptive}}({x})=\alpha\,\text{Mish}(\beta{x}+\gamma),

    where :math:`\alpha`, :math:`\beta`, and :math:`\gamma` are trainable
    parameters controlling output scaling, input scaling, and input shifting,
    respectively.

    The Mish function is defined elementwise as:

    .. math::
        \text{Mish}(x) = x * \text{Tanh}(x)

    .. seealso::

        **Original reference**: Godfrey, L. B., Gashler, M. S. (2015).
        *A continuum among logarithmic, linear, and exponential functions,
        and its potential to improve generalization in neural networks.*
        7th international joint conference on knowledge discovery, knowledge
        engineering and knowledge management (IC3K), Vol. 1.
        DOI: `arXiv preprint arXiv:1602.01321.
        <https://arxiv.org/abs/1602.01321>`_.

        **Original reference**: Jagtap, A. D., Karniadakis, G. E. (2020).
        *Adaptive activation functions accelerate convergence in deep and
        physics-informed neural networks*.
        Journal of Computational Physics, 404.
        DOI: `JCP 10.1016 <https://doi.org/10.1016/j.jcp.2019.109136>`_.
    """

    def __init__(self, alpha=None, beta=None, gamma=None, fixed=None):
        """
        Initialization of the :class:`AdaptiveMish` class.

        :param alpha: The output scaling parameter of the adaptive function.
            If ``None``, it is initialized to ``1``. Default is ``None``.
        :type alpha: int | float | complex
        :param beta: The input scaling parameter of the adaptive function.
            If ``None``, it is initialized to ``1``. Default is ``None``.
        :type beta: int | float | complex
        :param gamma: The input shifting parameter of the adaptive function.
            If ``None``, it is initialized to ``0``. Default is ``None``.
        :type gamma: int | float | complex
        :param fixed: The names of parameters to keep fixed during training.
            These parameters will not be optimized and will have
            ``requires_grad=False``. Available options are ``"alpha"``,
            ``"beta"``, and ``"gamma"``. If ``None``, all parameters are
            trainable. Default is ``None``.
        :type fixed: str | list[str]
        :raises ValueError: If alpha, when provided, is not a number.
        :raises ValueError: If beta, when provided, is not a number.
        :raises ValueError: If gamma, when provided, is not a number.
        :raises ValueError: If fixed, when provided, is neither a string nor a
            list of strings.
        :raises ValueError: If fixed contains invalid parameter names.
        """
        super().__init__(alpha, beta, gamma, fixed)
        self.func = torch.nn.Mish()
