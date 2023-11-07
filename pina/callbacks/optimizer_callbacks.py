'''PINA Callbacks Implementations'''

from pytorch_lightning.callbacks import Callback
import torch
from ..utils import check_consistency


class SwitchOptimizer(Callback):
    """
    PINA implementation of a Lightining Callback to switch
    optimizer during training. The rouutine can be used to
    try multiple optimizers during the training, without the
    need to stop training.
    """

    def __init__(self, new_optimizers, new_optimizers_kwargs, epoch_switch):
        """
        SwitchOptimizer is a routine for switching optimizer during training.

        :param torch.optim.Optimizer | list new_optimizers: The model optimizers to
            switch to. It must be a list of :class:`torch.optim.Optimizer` or list of
            :class:`torch.optim.Optimizer` for multiple model solvers.
        :param dict| list new_optimizers: The model optimizers keyword arguments to
            switch use. It must be a dict or list of dict for multiple optimizers.
        :param int epoch_switch: Epoch for switching optimizer.
        """
        super().__init__()

        # check type consistency
        check_consistency(new_optimizers, torch.optim.Optimizer, subclass=True)
        check_consistency(new_optimizers_kwargs, dict)
        check_consistency(epoch_switch, int)

        if epoch_switch < 1:
            raise ValueError('epoch_switch must be greater than one.')

        if not isinstance(new_optimizers, list):
            new_optimizers = [new_optimizers]
            new_optimizers_kwargs = [new_optimizers_kwargs]
        len_optimizer = len(new_optimizers)
        len_optimizer_kwargs = len(new_optimizers_kwargs)

        if len_optimizer_kwargs != len_optimizer:
            raise ValueError('You must define one dictionary of keyword'
                             ' arguments for each optimizers.'
                             f' Got {len_optimizer} optimizers, and'
                             f' {len_optimizer_kwargs} dicitionaries')

        # save new optimizers
        self._new_optimizers = new_optimizers
        self._new_optimizers_kwargs = new_optimizers_kwargs
        self._epoch_switch = epoch_switch

    def on_train_epoch_start(self, trainer, __):
        if trainer.current_epoch == self._epoch_switch:
            optims = []
            for idx, (optim, optim_kwargs) in enumerate(
                    zip(self._new_optimizers, self._new_optimizers_kwargs)):
                optims.append(
                    optim(trainer._model.models[idx].parameters(),
                          **optim_kwargs))

            trainer.optimizers = optims
