'''PINA Callbacks Implementations'''

from pytorch_lightning.callbacks import Callback
import torch
from ..utils import check_consistency


class SwitchOptimizer(Callback):

    def __init__(self, new_optimizers, new_optimizers_kwargs, epoch_switch):
        """
        PINA Implementation of a Lightning Callback to switch optimizer during training.

        This callback allows for switching between different optimizers during training, enabling
        the exploration of multiple optimization strategies without the need to stop training.

        :param new_optimizers: The model optimizers to switch to. Can be a single 
                            :class:`torch.optim.Optimizer` or a list of them for multiple model solvers.
        :type new_optimizers: torch.optim.Optimizer | list
        :param new_optimizers_kwargs: The keyword arguments for the new optimizers. Can be a single dictionary
                                    or a list of dictionaries corresponding to each optimizer.
        :type new_optimizers_kwargs: dict | list
        :param epoch_switch: The epoch at which to switch to the new optimizer.
        :type epoch_switch: int

        :raises ValueError: If `epoch_switch` is less than 1 or if there is a mismatch in the number of 
                            optimizers and their corresponding keyword argument dictionaries.

        Example:
            >>> switch_callback = SwitchOptimizer(new_optimizers=[optimizer1, optimizer2],
            >>>                                  new_optimizers_kwargs=[{'lr': 0.001}, {'lr': 0.01}],
            >>>                                  epoch_switch=10)
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
        """
        Callback function to switch optimizer at the start of each training epoch.

        :param trainer: The trainer object managing the training process.
        :type trainer: pytorch_lightning.Trainer
        :param _: Placeholder argument (not used).

        :return: None
        :rtype: None
        """
        if trainer.current_epoch == self._epoch_switch:
            optims = []
            for idx, (optim, optim_kwargs) in enumerate(
                    zip(self._new_optimizers, self._new_optimizers_kwargs)):
                optims.append(
                    optim(trainer._model.models[idx].parameters(),
                          **optim_kwargs))

            trainer.optimizers = optims
