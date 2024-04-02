""" Module for adaptive functions. """

import torch
from pina.utils import check_consistency


class AdaptiveActivationFunction(torch.nn.Module):
    r"""
    The :class:`~pina.model.layers.adaptive_func.AdaptiveActivationFunction`
    class makes a :class:`torch.nn.Module` activation function into an adaptive
    trainable activation function.

    Given a function :math:`f:\mathbb{R}^n\rightarrow\mathbb{R}^m`, the adaptive
    function :math:`f_{\text{adaptive}}:\mathbb{R}^n\rightarrow\mathbb{R}^m`
    is defined as:

    .. math::
        f_{\text{adaptive}}(\mathbf{x}) = \alpha\,f(\beta\mathbf{x}+\gamma),

    where :math:`\alpha,\,\beta,\,\gamma` are trainable parameters.

    :Example:
        >>> import torch
        >>> from pina.model.layers import AdaptiveActivationFunction
        >>>
        >>> # simple adaptive function with all trainable parameters
        >>> AdaptiveTanh = AdaptiveActivationFunction(torch.nn.Tanh())
        >>> AdaptiveTanh(torch.rand(3))
        tensor([0.1084, 0.3931, 0.7294], grad_fn=<MulBackward0>)
        >>> AdaptiveTanh.alpha
        Parameter containing:
        tensor(1., requires_grad=True)
        >>>
        >>> # simple adaptive function with trainable parameters fixed alpha
        >>> AdaptiveTanh = AdaptiveActivationFunction(torch.nn.Tanh(),
        ...                                           fixed=['alpha'])
        >>> AdaptiveTanh.alpha
        tensor(1.)
        >>> AdaptiveTanh.beta
        Parameter containing:
        tensor(1., requires_grad=True)
        >>>

    .. seealso::

        **Original reference**: Godfrey, Luke B., and Michael S. Gashler.
        *A continuum among logarithmic, linear, and exponential functions,
        and its potential to improve generalization in neural networks.*
        2015 7th international joint conference on knowledge discovery,
        knowledge engineering and knowledge management (IC3K).
        Vol. 1. IEEE, 2015. DOI: `arXiv preprint arXiv:1602.01321.
        <https://arxiv.org/abs/1602.01321>`_.

    """

    def __init__(self, func, alpha=None, beta=None, gamma=None, fixed=None):
        """
        Initializes the AdaptiveActivationFunction module.

        :param callable func: The original collable function. It could be an
            initialized :meth:`torch.nn.Module`, or a python callable function.
        :param float | complex alpha: Scaling parameter alpha.
            Defaults to ``None``. When ``None`` is passed,
            the variable is initialized to 1.
        :param float | complex beta: Scaling parameter beta.
            Defaults to ``None``. When ``None`` is passed,
            the variable is initialized to 1.
        :param float | complex gamma: Shifting parameter gamma.
            Defaults to ``None``. When ``None`` is passed,
            the variable is initialized to 1.
        :param list fixed: List of parameters to fix during training,
            i.e. not optimized (``requires_grad`` set to ``False``).
            Options are ['alpha', 'beta', 'gamma']. Defaults to None.
        """
        super().__init__()

        # see if there are fixed variables
        if fixed is not None:
            check_consistency(fixed, str)
            if not all(key in ["alpha", "beta", "gamma"] for key in fixed):
                raise TypeError(
                    "Fixed keys must be in [`alpha`, `beta`, `gamma`]."
                )

        # initialize alpha, beta, gamma if they are None
        if alpha is None:
            alpha = 1.0
        if beta is None:
            beta = 1.0
        if gamma is None:
            gamma = 0.0

        # checking consistency
        check_consistency(alpha, (float, complex))
        check_consistency(beta, (float, complex))
        check_consistency(gamma, (float, complex))
        if not callable(func):
            raise ValueError("Function must be a callable function.")

        # registering as tensors
        alpha = torch.tensor(alpha, requires_grad=False)
        beta = torch.tensor(beta, requires_grad=False)
        gamma = torch.tensor(gamma, requires_grad=False)

        # setting not fixed variables as torch.nn.Parameter with gradient
        # registering the buffer for the one which are fixed, buffers by
        # default are saved alongside trainable parameters
        if "alpha" not in (fixed or []):
            self._alpha = torch.nn.Parameter(alpha, requires_grad=True)
        else:
            self.register_buffer("alpha", alpha)

        if "beta" not in (fixed or []):
            self._beta = torch.nn.Parameter(beta, requires_grad=True)
        else:
            self.register_buffer("beta", beta)

        if "gamma" not in (fixed or []):
            self._gamma = torch.nn.Parameter(gamma, requires_grad=True)
        else:
            self.register_buffer("gamma", gamma)

        # registering function
        self._func = func

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        return self.alpha * (self._func(self.beta * x + self.gamma))

    @property
    def alpha(self):
        """
        The alpha variable
        """
        return self._alpha

    @property
    def beta(self):
        """
        The alpha variable
        """
        return self._beta

    @property
    def gamma(self):
        """
        The alpha variable
        """
        return self._gamma
