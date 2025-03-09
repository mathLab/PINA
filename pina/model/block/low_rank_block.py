"""Module for Averaging Neural Operator Layer class."""

import torch

from ...utils import check_consistency


class LowRankBlock(torch.nn.Module):
    r"""
    The PINA implementation of the inner layer of the Averaging Neural Operator.

    The operator layer performs an affine transformation where the convolution
    is approximated with a local average. Given the input function
    :math:`v(x)\in\mathbb{R}^{\rm{emb}}` the layer computes
    the operator update :math:`K(v)` as:

    .. math::
        K(v) = \sigma\left(Wv(x) + b + \sum_{i=1}^r \langle
        \psi^{(i)} , v(x) \rangle \phi^{(i)} \right)

    where:

    *   :math:`\mathbb{R}^{\rm{emb}}` is the embedding (hidden) size
        corresponding to the ``hidden_size`` object
    *   :math:`\sigma` is a non-linear activation, corresponding to the
        ``func`` object
    *   :math:`W\in\mathbb{R}^{\rm{emb}\times\rm{emb}}` is a tunable matrix.
    *   :math:`b\in\mathbb{R}^{\rm{emb}}` is a tunable bias.
    *   :math:`\psi^{(i)}\in\mathbb{R}^{\rm{emb}}` and
        :math:`\phi^{(i)}\in\mathbb{R}^{\rm{emb}}` are :math:`r` a low rank
        basis functions mapping.
    *   :math:`b\in\mathbb{R}^{\rm{emb}}` is a tunable bias.

    .. seealso::

        **Original reference**: Kovachki, N., Li, Z., Liu, B.,
        Azizzadenesheli, K., Bhattacharya, K., Stuart, A., & Anandkumar, A.
        (2023). *Neural operator: Learning maps between function
        spaces with applications to PDEs*. Journal of Machine Learning
        Research, 24(89), 1-97.

    """

    def __init__(
        self,
        input_dimensions,
        embedding_dimenion,
        rank,
        inner_size=20,
        n_layers=2,
        func=torch.nn.Tanh,
        bias=True,
    ):
        """
        :param int input_dimensions: The number of input components of the
            model.
            Expected tensor shape of the form :math:`(*, d)`, where *
            means any number of dimensions including none,
            and :math:`d` the ``input_dimensions``.
        :param int embedding_dimenion: Size of the embedding dimension of the
            field.
        :param int rank: The rank number of the basis approximation components
            of the model. Expected tensor shape of the form :math:`(*, 2d)`,
            where * means any number of dimensions including none,
            and :math:`2d` the ``rank`` for both basis functions.
        :param int inner_size: Number of neurons in the hidden layer(s) for the
            basis function network. Default is 20.
        :param int n_layers: Number of hidden layers. for the
            basis function network. Default is 2.
        :param func: The activation function to use for the
            basis function network. If a single
            :class:`torch.nn.Module` is passed, this is used as
            activation function after any layers, except the last one.
            If a list of Modules is passed,
            they are used as activation functions at any layers, in order.
        :param bool bias: If ``True`` the MLP will consider some bias for the
            basis function network.
        """
        super().__init__()
        # Avoid circular import. I need to import FeedForward here
        # to avoid circular import with FeedForward itself.
        # pylint: disable=import-outside-toplevel
        from ..feed_forward import FeedForward

        # Assignment (check consistency inside FeedForward)
        self._basis = FeedForward(
            input_dimensions=input_dimensions,
            output_dimensions=2 * rank * embedding_dimenion,
            inner_size=inner_size,
            n_layers=n_layers,
            func=func,
            bias=bias,
        )
        self._nn = torch.nn.Linear(embedding_dimenion, embedding_dimenion)

        check_consistency(rank, int)
        self._rank = rank
        self._func = func()

    def forward(self, x, coords):
        r"""
        Forward pass of the layer, it performs an affine transformation of
        the field, and a low rank approximation by
        doing a dot product of the basis
        :math:`\psi^{(i)}` with the filed vector :math:`v`, and use this
        coefficients to expand :math:`\phi^{(i)}` evaluated in the
        spatial input :math:`x`.

        :param torch.Tensor x: The input tensor for performing the
            computation. It expects a tensor :math:`B \times N \times D`,
            where :math:`B` is the batch_size, :math:`N` the number of points
            in the mesh, :math:`D` the dimension of the problem. In particular
            :math:`D` is the codomain of the function :math:`v`. For example
            a scalar function has :math:`D=1`, a 4-dimensional vector function
            :math:`D=4`.
        :param torch.Tensor coords: The coordinates in which the field is
            evaluated for performing the  computation. It expects a
            tensor :math:`B \times N \times d`,
            where :math:`B` is the batch_size, :math:`N` the number of points
            in the mesh, :math:`D` the dimension of the domain.
        :return: The output tensor obtained from Average Neural Operator Block.
        :rtype: torch.Tensor
        """
        # extract basis
        coords = coords.as_subclass(torch.Tensor)
        basis = self._basis(coords)
        # reshape [B, N, D, 2*rank]
        shape = list(basis.shape[:-1]) + [-1, 2 * self.rank]
        basis = basis.reshape(shape)
        # divide
        psi = basis[..., : self.rank]
        phi = basis[..., self.rank :]
        # compute dot product
        coeff = torch.einsum("...dr,...d->...r", psi, x)
        # expand the basis
        expansion = torch.einsum("...r,...dr->...d", coeff, phi)
        # apply linear layer and return
        return self._func(self._nn(x) + expansion)

    @property
    def rank(self):
        """
        The basis rank.
        """
        return self._rank
