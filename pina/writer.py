""" Module for plotting. """
import matplotlib.pyplot as plt
import numpy as np
import torch

from pina import LabelTensor


class Writer:
    """
    Implementation of a writer class, for textual output.
    """

    def __init__(self,
                 frequency_print=10,
                 header='any') -> None:

        """
        The constructor of the class.

        :param int frequency_print: the frequency in epochs of printing.
        :param ['any', 'begin', 'none'] header: the header of the output.
        """

        self._frequency_print = frequency_print
        self._header = header


    def header(self, trainer):
        """
        The method for printing the header.
        """
        header = []
        for condition_name in trainer.problem.conditions:
            header.append(f'{condition_name}')

        return header

    def write_loss(self, trainer):
        """
        The method for writing the output.
        """
        pass


    def write_loss_in_loop(self, trainer, loss):
        """
        The method for writing the output within the training loop.

        :param pina.trainer.Trainer trainer: the trainer object.
        """

        if trainer.trained_epoch % self._frequency_print == 0:
            print(f'Epoch {trainer.trained_epoch:05d}: {loss.item():.5e}')
