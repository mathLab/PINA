""" Solver module. """

import lightning.pytorch as pl
from .utils import check_consistency
from .dataset import DummyLoader
from .solvers.solver import SolverInterface

class Trainer(pl.Trainer):

    def __init__(self, solver, **kwargs):
        super().__init__(**kwargs)

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
        self._create_or_update_loader()

    # this method is used here because is resampling is needed
    # during training, there is no need to define to touch the
    # trainer dataloader, just call the method.
    def _create_or_update_loader(self):
        # get accellerator
        device = self._accelerator_connector._accelerator_flag
        self._loader = DummyLoader(self._model.problem.input_pts, device) 

    def train(self, **kwargs): # TODO add kwargs and lightining capabilities
        return super().fit(self._model, self._loader, **kwargs)
    
    @property
    def solver(self):
        """
        Returning trainer solver.
        """
        return self._model