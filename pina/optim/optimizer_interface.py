""" Module for PINA Optimizer """

from abc import ABCMeta, abstractmethod


class Optimizer(metaclass=ABCMeta):  # TODO improve interface

    @property
    @abstractmethod
    def instance(self):
        pass

    @abstractmethod
    def hook(self):
        pass