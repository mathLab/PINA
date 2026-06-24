"""Module for the Adaptive Function base class."""

import torch
from pina._src.core.utils import check_consistency
from pina._src.adaptive_function.adaptive_function_interface import (
    AdaptiveFunctionInterface,
)


class BaseAdaptiveFunction(torch.nn.Module, AdaptiveFunctionInterface):
    r"""
    Base class for all adaptive functions, implementing common functionality.

    This class extends a standard :class:`torch.nn.Module` activation function
    into a trainable adaptive form. It implements the common mechanism used to
    scale and shift both the input and the output of a given activation
    function.

    Given a function :math:`f:\mathbb{R}^n\rightarrow\mathbb{R}^m`, the adaptive
    function :math:`f_{\text{adaptive}}:\mathbb{R}^n\rightarrow\mathbb{R}^m`
    is defined as:

    .. math::
        f_{\text{adaptive}}(\mathbf{x}) = \alpha\,f(\beta\mathbf{x}+\gamma),

    where :math:`\alpha`, :math:`\beta`, and :math:`\gamma` are learnable
    parameters controlling output scaling, input scaling, and input shifting,
    respectively.

    All specific adaptive functions should inherit from this class and implement
    the abstract methods declared in the interface.

    This class is not meant to be instantiated directly.

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
        Initialization of the :class:`BaseAdaptiveFunction` class.

        :param alpha: The output scaling parameter of the adaptive function.
            If ``None``, it is initialized to ``1``. Default is ``None``.
        :type alpha: int | float
        :param beta: The input scaling parameter of the adaptive function.
            If ``None``, it is initialized to ``1``. Default is ``None``.
        :type beta: int | float
        :param gamma: The input shifting parameter of the adaptive function.
            If ``None``, it is initialized to ``0``. Default is ``None``.
        :type gamma: int | float
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
        super().__init__()

        # Set default values for alpha, beta, gamma if they are None
        alpha = 1.0 if alpha is None else alpha
        beta = 1.0 if beta is None else beta
        gamma = 0.0 if gamma is None else gamma

        # Check consistency
        check_consistency(alpha, (int, float))
        check_consistency(beta, (int, float))
        check_consistency(gamma, (int, float))

        # Process fixed parameters
        if fixed is not None:
            check_consistency(fixed, str)
            fixed = {fixed} if isinstance(fixed, str) else set(fixed)
        else:
            fixed = set()

        # Validate fixed parameter names
        invalid_names = fixed - {"alpha", "beta", "gamma"}
        if invalid_names:
            raise ValueError(
                f"Invalid fixed parameter name(s): {sorted(invalid_names)}. "
                "Available options are 'alpha', 'beta', and 'gamma'."
            )

        # Register either a trainable parameter or a fixed buffer
        def _register_adaptive_param(name, value):
            """
            Helper function to register an adaptive parameter as either a
            trainable parameter or a fixed buffer, depending on whether it is
            specified in the ``fixed`` argument.
            """
            # Convert value to tensor
            tensor = torch.tensor(value, dtype=torch.float32)

            # Register as buffer if fixed, otherwise as parameter
            if name in fixed:
                self.register_buffer(f"_{name}", tensor)
            else:
                setattr(self, f"_{name}", torch.nn.Parameter(tensor))

        # Register parameters
        _register_adaptive_param("alpha", alpha)
        _register_adaptive_param("beta", beta)
        _register_adaptive_param("gamma", gamma)

        # Initialize the adaptive function to None, to be set by subclasses
        self._func = None

    def forward(self, x):
        """
        Compute the transformation of the adaptive function on the input.

        :param x: The input tensor to evaluate the adaptive function.
        :type x: torch.Tensor | LabelTensor
        :raises RuntimeError: If the adaptive function has not been set.
        :raises RuntimeError: If the adaptive function is not callable.
        :return: The output of the adaptive function.
        :rtype: torch.Tensor | LabelTensor
        """
        # Raise an error if the adaptive function has not been set
        if self._func is None:
            raise RuntimeError("The adaptive function has not been set.")

        # Raise an error if the adaptive function is not callable
        if not callable(self._func):
            raise RuntimeError("The adaptive function is not callable.")

        return self.alpha * (self._func(self.beta * x + self.gamma))

    @property
    def alpha(self):
        """
        The output scaling parameter of the adaptive function.

        :return: The alpha parameter.
        :rtype: torch.nn.Parameter | torch.Tensor
        """
        return self._alpha

    @property
    def beta(self):
        """
        The input scaling parameter of the adaptive function.

        :return: The beta parameter.
        :rtype: torch.nn.Parameter | torch.Tensor
        """
        return self._beta

    @property
    def gamma(self):
        """
        The input shifting parameter of the adaptive function.

        :return: The gamma parameter.
        :rtype: torch.nn.Parameter | torch.Tensor
        """
        return self._gamma
