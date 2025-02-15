__all__ = [
    'LpLoss',
    'PowerLoss',
    'weightningInterface',
    'LossInterface'
]

from .loss_interface import LossInterface
from .power_loss import PowerLoss
from .lp_loss import LpLoss
from .weightning_interface import weightningInterface