"""
Kernel Neural Operator Module.
"""

import torch
from pina.utils import check_consistency


class KernelNeuralOperator(torch.nn.Module):
    r"""
    Base class for composing Neural Operators with integral kernels.

    This is a base class for composing neural operators with multiple
    integral kernels. All neural operator models defined in PINA inherit
    from this class. The structure is inspired by the work of Kovachki, N.
    et al. see Figure 2 of the reference for extra details. The Neural
    Operators inheriting from this class can be written as:

    .. math::
        G_\theta  := P \circ K_m \circ \cdot \circ K_1 \circ L

    where:

    *   :math:`G_\theta: \mathcal{A}\subset \mathbb{R}^{\rm{in}} \rightarrow
        \mathcal{D}\subset \mathbb{R}^{\rm{out}}` is the neural operator
        approximation of the unknown real operator :math:`G`, that is
        :math:`G \approx G_\theta`
    *   :math:`L: \mathcal{A}\subset \mathbb{R}^{\rm{in}} \rightarrow
        \mathbb{R}^{\rm{emb}}` is a lifting operator mapping the input
        from its domain :math:`\mathcal{A}\subset \mathbb{R}^{\rm{in}}`
        to its embedding dimension :math:`\mathbb{R}^{\rm{emb}}`
    *   :math:`\{K_i : \mathbb{R}^{\rm{emb}} \rightarrow
        \mathbb{R}^{\rm{emb}} \}_{i=1}^m` are :math:`m` integral kernels
        mapping each hidden representation to the next one.
    *   :math:`P : \mathbb{R}^{\rm{emb}} \rightarrow  \mathcal{D}\subset
        \mathbb{R}^{\rm{out}}` is a projection operator mapping the hidden
        representation to the output function.

    .. seealso::

        **Original reference**: Kovachki, N., Li, Z., Liu, B.,
        Azizzadenesheli, K., Bhattacharya, K., Stuart, A., & Anandkumar, A.
        (2023). *Neural operator: Learning maps between function
        spaces with applications to PDEs*. Journal of Machine Learning
        Research, 24(89), 1-97.
    """

    def __init__(self, lifting_operator, integral_kernels, projection_operator):
        """
        :param torch.nn.Module lifting_operator: The lifting operator
            mapping the input to its hidden dimension.
        :param torch.nn.Module integral_kernels: List of integral kernels
            mapping each hidden representation to the next one.
        :param torch.nn.Module projection_operator: The projection operator
            mapping the hidden representation to the output function.
        """

        super().__init__()

        self._lifting_operator = lifting_operator
        self._integral_kernels = integral_kernels
        self._projection_operator = projection_operator

    @property
    def lifting_operator(self):
        """
        The lifting operator property.
        """
        return self._lifting_operator

    @lifting_operator.setter
    def lifting_operator(self, value):
        """
        The lifting operator setter

        :param torch.nn.Module value: The lifting operator torch module.
        """
        check_consistency(value, torch.nn.Module)
        self._lifting_operator = value

    @property
    def projection_operator(self):
        """
        The projection operator property.
        """
        return self._projection_operator

    @projection_operator.setter
    def projection_operator(self, value):
        """
        The projection operator setter

        :param torch.nn.Module value: The projection operator torch module.
        """
        check_consistency(value, torch.nn.Module)
        self._projection_operator = value

    @property
    def integral_kernels(self):
        """
        The integral kernels operator property.
        """
        return self._integral_kernels

    @integral_kernels.setter
    def integral_kernels(self, value):
        """
        The integral kernels operator setter

        :param torch.nn.Module value: The integral kernels operator torch
            module.
        """
        check_consistency(value, torch.nn.Module)
        self._integral_kernels = value

    def forward(self, x):
        r"""
        Forward computation for Base Neural Operator. It performs a
        lifting of the input by the ``lifting_operator``.
        Then different layers integral kernels are applied using
        ``integral_kernels``. Finally the output is projected
        to the final dimensionality by the ``projection_operator``.

        :param torch.Tensor x: The input tensor for performing the
            computation. It expects a tensor :math:`B \times N \times D`,
            where :math:`B` is the batch_size, :math:`N` the number of points
            in the mesh, :math:`D` the dimension of the problem. In particular
            :math:`D` is the number of spatial/paramtric/temporal variables
            plus the field variables. For example for 2D problems with 2
            output\ variables :math:`D=4`.
        :return: The output tensor obtained from the NO.
        :rtype: torch.Tensor
        """
        x = self.lifting_operator(x)
        x = self.integral_kernels(x)
        x = self.projection_operator(x)
        return x
