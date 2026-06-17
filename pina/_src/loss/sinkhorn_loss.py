"""Module for the SinkhornLoss class."""

import torch
from pina._src.loss.base_dual_loss import BaseDualLoss
from pina._src.core.utils import check_consistency, check_positive_integer


class SinkhornLoss(BaseDualLoss):
    r"""
    Implementation of the Sinkhorn Loss based on regularized optimal transport.
    It measures the regularized Wasserstein distance between the empirical
    distributions represented by ``input`` (with :math:`N` samples) and
    ``target`` (with :math:`M` samples), each in :math:`\mathbb{R}^D`.

    The loss solves the entropy-regularized optimal transport problem:

    .. math::
        W_\varepsilon(\mu, \nu) = \min_{\pi \in \Pi(\mu, \nu)}
        \langle C, \pi \rangle - \varepsilon H(\pi),

    where :math:`C_{ij} = \|x_i - y_j\|_2^p` is the cost matrix,
    :math:`H(\pi) = -\sum_{ij} \pi_{ij} \log \pi_{ij}` is the entropy of
    the transport plan, and :math:`\varepsilon > 0` is the regularization
    strength. The dual objective recovered by the Sinkhorn iterations is:

    .. math::
        W_\varepsilon = \langle a, f^* \rangle + \langle b, g^* \rangle,

    where :math:`a` and :math:`b` are uniform probability weights over the
    :math:`N` and :math:`M` samples respectively, and :math:`f^*, g^*` are
    the optimal dual potentials computed via log-space Sinkhorn iterations.

    If ``reduction`` is set to ``"mean"`` or ``"sum"``, the scalar transport
    cost is aggregated accordingly (the output is always a scalar, so both
    reductions are equivalent):

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{``mean''} \\
            \operatorname{sum}(L),  & \text{if reduction} = \text{``sum''}
        \end{cases}

    .. note::
        Unlike pointwise losses, the Sinkhorn loss operates on entire empirical
        distributions, so the output is always a scalar regardless of the
        number of samples. The ``reduction`` parameter is retained for API
        consistency.

    .. note::
        Smaller values of ``eps`` approximate the true Wasserstein distance
        more closely but may require more iterations to converge.

    .. note::
        The algorithm is taken from "Sinkhorn AutoEncoders", arXiv:1810.01118.
    """

    def __init__(self, p=2, eps=0.1, max_iter=100, reduction="mean"):
        """
        Initialization of the :class:`SinkhornLoss` class.

        :param int p: Exponent of the cost function :math:`\|x_i - y_j\|_2^p`.
            Default is ``2``.
        :param float eps: Entropy regularization strength
            :math:`\varepsilon > 0`. Larger values yield smoother transport
            plans. Default is ``0.1``.
        :param int max_iter: Number of Sinkhorn iterations. Default is ``100``.
        :param str reduction: The reduction method to aggregate the scalar loss.
            Available options include: ``"none"``, ``"mean"``, ``"sum"``.
            Default is ``"mean"``.
        :raises ValueError: If ``p`` is not a numeric value.
        :raises ValueError: If ``eps`` is not a positive float.
        :raises AssertionError: If ``max_iter`` is not a strictly positive int.
        """
        super().__init__(reduction=reduction)

        check_consistency(p, (int, float))
        check_consistency(eps, float)
        if eps <= 0:
            raise ValueError(
                f"eps must be a strictly positive float, got {eps}."
            )
        check_positive_integer(max_iter, strict=True)

        self.p = p
        self.eps = eps
        self.max_iter = max_iter

    def forward(self, input, target):
        """
        Forward method of the loss function.

        :param torch.Tensor input: Input tensor of shape :math:`(N, D)`.
        :param torch.Tensor target: Target tensor of shape :math:`(M, D)`.
        :return: Sinkhorn loss value.
        :rtype: torch.Tensor
        """
        n = input.shape[0]
        m = target.shape[0]

        a = input.new_full((n,), 1.0 / n)
        b = target.new_full((m,), 1.0 / m)

        # Cost matrix C[i,j] = ||x_i - y_j||_2^p, shape (N, M)
        diff = input.unsqueeze(1) - target.unsqueeze(0)  # (N, M, D)
        C = torch.linalg.norm(diff, ord=2, dim=-1).pow(self.p)  # (N, M)

        # Log-space Sinkhorn iterations for numerical stability
        log_a = a.log()
        log_b = b.log()
        f = torch.zeros(n, dtype=input.dtype, device=input.device)
        g = torch.zeros(m, dtype=target.dtype, device=target.device)

        for _ in range(self.max_iter):
            # f_i = eps * (log a_i - logsumexp_j ((g_j - C_ij) / eps))
            f = self.eps * (
                log_a
                - torch.logsumexp((g.unsqueeze(0) - C) / self.eps, dim=1)
            )
            # g_j = eps * (log b_j - logsumexp_i ((f_i - C_ij) / eps))
            g = self.eps * (
                log_b
                - torch.logsumexp((f.unsqueeze(1) - C) / self.eps, dim=0)
            )

        loss = (a * f).sum() + (b * g).sum()
        return self._reduction(loss.unsqueeze(0))
