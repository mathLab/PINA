""" Solver module. """

import lightning.pytorch as pl
from .utils import check_consistency
from .dataset import DummyLoader
from .solvers.solver import SolverInterface
from pytorch_lightning.loggers import TensorBoardLogger


class Trainer(pl.Trainer):

    def __init__(self, solver, **kwargs):
        super().__init__(**kwargs)
        
        # get accellerator
        device = self._accelerator_connector._accelerator_flag

        # check inheritance consistency for solver
        check_consistency(solver, SolverInterface)
        self._model = solver

        # create dataloader
        if solver.problem.have_sampled_points is False:
            raise RuntimeError(f'Input points in {solver.problem.not_sampled_points} '
                               'training are None. Please '
                               'sample points in your problem by calling '
                               'discretise_domain function before train '
                               'in the provided locations.')

        # TODO: make a better dataloader for train
        self._loader = DummyLoader(solver.problem.input_pts, device) 

        # handling loggers
        self._pina_logger = PinaLogger(self.loggers)

        # self._logger = self._logger_connector.trainer.loggers

        # Initialize TensorBoardLogger
        self.logger = TensorBoardLogger('logs/', name='my_experiment')

    def train(self):  # TODO add kwargs and lightining capabilities
        self.logger.log_hyperparams(self._model.hparams)
        self.logger.log_graph(self._model)

        super().fit(self._model, self._loader)

        # self.logger.experiment.close()
        # self.close()
