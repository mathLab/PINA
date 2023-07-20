
from abc import abstractmethod, ABCMeta
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities import rank_zero_only


class PINALogger(Logger, metaclass=ABCMeta):
    """ Base class wrapper for all loggers in Lightning """

    @abstractmethod
    def call(self, model):
        """Lightning calls this hook with the trainer"""
        pass


# import lightning.pytorch as pl

# class PinaLogger(object):
#     """
#     PINA logger wrapper
#     """

#     def __init__(self, loggers) -> None:
#         self.loggers = loggers

#         if len(self.loggers) > 1:
#             raise NotImplementedError(
#                 'Currently implementing one logger at a time.')

#     def log_graph(self, model):
#         self.loggers[0].log_graph(model=model)
