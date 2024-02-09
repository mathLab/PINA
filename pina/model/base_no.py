import torch
from pina.utils import check_consistency

class BaseNO(torch.nn.Module):
    def __init__(self, lifting_operator, integral_kernels, projection_operator):
        r"""
        Base class for composing Neural Operators with integral kernels.

        This is a base class for composing neural operators with multiple
        integral kernels. All neural operator models defined in PINA inherit
        from this class. The structure is inspired by the work of Kovachki, N.
        et al. see Figure 2 of the reference for extra details. The Neural
        Operators inheriting from this class can be written as:
        $$ G_\theta  := P \circ K_L \circ\cdot\circ K_1 \circ L,$$
        where $L$ is a lifting operator mapping the input to its hidden
        dimension. The $\{K_i\}_{i=1}^L$ are integral kernels mapping
        each hidden representation to the next one. Finally $P$ is a projection
        operator mapping the hidden representation to the output function.

        .. seealso::

            **Original reference**: Kovachki, N., Li, Z., Liu, B., Azizzadenesheli,
            K., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2023). *Neural
            operator: Learning maps between function spaces with applications
            to PDEs*. Journal of Machine Learning Research, 24(89), 1-97.

        :param lifting_operator: The lifting operator mapping the input to its hidden dimension.
        :type lifting_operator: torch.nn.Module
        :param integral_kernels: List of integral kernels mapping each hidden representation to the next one.
        :type integral_kernels: torch.nn.Module
        :param projection_operator: The projection operator mapping the hidden representation to the output function.
        :type projection_operator: torch.nn.Module
        """

        super().__init__()

        self._lifting_operator = lifting_operator
        self._integral_kernels = integral_kernels
        self._projection_operator = projection_operator

    @property
    def lifting_operator(self):
        return self._lifting_operator 

    @lifting_operator.setter
    def lifting_operator(self, value):
        check_consistency(value, torch.nn.Module)
        self._lifting_operator = value

    @property
    def projection_operator(self):
        return self._projection_operator 

    @projection_operator.setter
    def projection_operator(self, value):
        check_consistency(value, torch.nn.Module)
        self._projection_operator = value

    @property
    def integral_kernels(self):
        return self._integral_kernels

    @integral_kernels.setter
    def integral_kernels(self, value):
        check_consistency(value, torch.nn.Module)
        self._integral_kernels = value


    def forward(self, x):
        """
        Forward computation for Base Neural Operator. It performs a
        lifting of the input by the ``lifting_operator``.
        Then different layers integral kernels are applied using
        ``integral_kernels``. Finally the output is projected
        to the final dimensionality by the ``projection_operator``.

        :param torch.Tensor x: The input tensor for performing the
            computation. It expects a tensor: $$[B \times \times N \times D]$$
        where $B$ is the batch_size, $N$ the number of points in the mesh,
        $D$ the dimension of the problem. In particular  $D$ is the number
        of spatial/paramtric/temporal variables + field variables.
        For example for 2D problems with 2 output variables $D=4$.

        :return: The output tensor obtained from the NO.
        :rtype: torch.Tensor
        """
        x = self.lifting_operator(x)
        x = self.integral_kernels(x)
        x = self.projection_operator(x)
        return x
