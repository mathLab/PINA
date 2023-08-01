import torch
import torch.nn as nn
from ...utils import check_consistency


class FourierBlock(nn.Module):
    """Fourier block base class. Implementation of a fourier block.

    .. seealso::

        **Original reference**: Li, Zongyi, et al.
        "Fourier neural operator for parametric partial
        differential equations." arXiv preprint
        arXiv:2010.08895 (2020)
        <https://arxiv.org/abs/2010.08895.pdf>`_.

    """

    def __init__(self):
        super().__init__()


    def forward(self, x):
        pass