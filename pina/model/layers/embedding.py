""" Periodic Boundary Embedding modulus. """

import torch
from pina.utils import check_consistency


class PeriodicBoundaryEmbedding(torch.nn.Module):
    r"""
    Imposing hard constraint periodic boundary conditions by embedding the
    input.

    A periodic function :math:`u:\mathbb{R}^{\rm{in}}
    \rightarrow\mathbb{R}^{\rm{out}}` periodic in the spatial
    coordinates :math:`\mathbf{x}` with periods :math:`\mathbf{L}` is such that:

    .. math::
        u(\mathbf{x})  = u(\mathbf{x} + n \mathbf{L})\;\;
        \forall n\in\mathbb{N}.

    The :meth:`PeriodicBoundaryEmbedding` augments the input such that the periodic conditons
    is guarantee. The input is augmented by the following formula:

    .. math::
        \mathbf{x} \rightarrow \tilde{\mathbf{x}} = \left[1,
        \cos\left(\frac{2\pi}{L_1} x_1 \right),
        \sin\left(\frac{2\pi}{L_1}x_1\right), \cdots,
        \cos\left(\frac{2\pi}{L_{\rm{in}}}x_{\rm{in}}\right),
        \sin\left(\frac{2\pi}{L_{\rm{in}}}x_{\rm{in}}\right)\right],

    where :math:`\text{dim}(\tilde{\mathbf{x}}) = 3\text{dim}(\mathbf{x})`.

    .. seealso::
        **Original reference**:
            1.  Dong, Suchuan, and Naxian Ni (2021). *A method for representing
                periodic functions and enforcing exactly periodic boundary
                conditions with deep neural networks*. Journal of Computational
                Physics 435, 110242.
                DOI: `10.1016/j.jcp.2021.110242.
                <https://doi.org/10.1016/j.jcp.2021.110242>`_
            2.  Wang, S., Sankaran, S., Wang, H., & Perdikaris, P. (2023). *An
                expert's guide to training physics-informed neural networks*.
                DOI: `arXiv preprint arXiv:2308.0846.
                <https://arxiv.org/abs/2308.08468>`_
    .. warning::
        The embedding is a truncated fourier expansion, and only ensures
        function PBC and not for its derivatives. Ensuring approximate
        periodicity in
        the derivatives of :math:`u` can be done, and extensive
        tests have shown (also in the reference papers) that this implementation
        can correctly compute the PBC on the derivatives up to the order
        :math:`\sim 2,3`, while it is not guarantee the periodicity for
        :math:`>3`. The PINA code is tested only for function PBC and not for
        its derivatives.
    """

    def __init__(self, input_dimension, periods, output_dimension=None):
        """
        :param int input_dimension: The dimension of the input tensor, it can
            be checked with `tensor.ndim` method.
        :param float | int | dict periods: The periodicity in each dimension for
            the input data. If ``float`` or ``int`` is passed,
            the period is assumed constant for all the dimensions of the data.
            If a ``dict`` is passed the `dict.values` represent periods,
            while the ``dict.keys`` represent the dimension where the
            periodicity is applied. The `dict.keys` can either be `int`
            if working with ``torch.Tensor`` or ``str`` if
            working with ``LabelTensor``.
        :param int output_dimension: The dimension of the output after the
            fourier embedding. If not ``None`` a ``torch.nn.Linear`` layer
            is applied to the fourier embedding output to match the desired
            dimensionality, default ``None``.
        """
        super().__init__()

        # check input consistency
        check_consistency(periods, (float, int, dict))
        check_consistency(input_dimension, int)
        if output_dimension is not None:
            check_consistency(output_dimension, int)
            self._layer = torch.nn.Linear(input_dimension * 3, output_dimension)
        else:
            self._layer = torch.nn.Identity()

        # checks on the periods
        if isinstance(periods, dict):
            if not all(
                isinstance(dim, (str, int)) and isinstance(period, (float, int))
                for dim, period in periods.items()
            ):
                raise TypeError(
                    "In dictionary periods, keys must be integers"
                    " or strings, and values must be float or int."
                )
            self._period = periods
        else:
            self._period = {k: periods for k in range(input_dimension)}

    def forward(self, x):
        """
        Forward pass to compute the periodic boundary conditions embedding.

        :param torch.Tensor x: Input tensor.
        :return: Fourier embeddings of the input.
        :rtype: torch.Tensor
        """
        omega = torch.stack(
            [
                torch.pi * 2.0 / torch.tensor([val], device=x.device)
                for val in self._period.values()
            ],
            dim=-1,
        )
        x = self._get_vars(x, list(self._period.keys()))
        return self._layer(
            torch.cat(
                [
                    torch.ones_like(x),
                    torch.cos(omega * x),
                    torch.sin(omega * x),
                ],
                dim=-1,
            )
        )

    def _get_vars(self, x, indeces):
        """
        Get variables from input tensor ordered by specific indeces.

        :param torch.Tensor x: The input tensor to extract.
        :param list[int] | list[str] indeces: List of indeces to extract.
        :return: The extracted tensor given the indeces.
        :rtype: torch.Tensor
        """
        if isinstance(indeces[0], str):
            try:
                return x.extract(indeces)
            except AttributeError:
                raise RuntimeError(
                    "Not possible to extract input variables from tensor."
                    " Ensure that the passed tensor is a LabelTensor or"
                    " pass list of integers to extract variables. For"
                    " more information refer to warning in the documentation."
                )
        elif isinstance(indeces[0], int):
            return x[..., indeces]
        else:
            raise RuntimeError(
                "Not able to extract right indeces for tensor."
                " For more information refer to warning in the documentation."
            )

    @property
    def period(self):
        """
        The period of the periodic function to approximate.
        """
        return self._period
