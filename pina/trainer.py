""" Solver module. """

import lightning.pytorch as pl
from .utils import check_consistency
from .dataset import DummyLoader
from .solver import SolverInterface

class Trainer(pl.Trainer):

    def __init__(self, solver, kwargs={}):
        super().__init__(**kwargs)
        
        # check inheritance consistency for solver
        check_consistency(solver, SolverInterface, 'Solver model')
        self._model = solver

        # create dataloader
        if solver.problem.have_sampled_points is False:
            raise RuntimeError('Input points for training is None. Please '
                               'sample points in your problem by calling '
                               'discretise_domain function before train.')
        
        # TODO: make a better dataloader for train
        self._loader = DummyLoader(solver.problem.input_pts) 


    def train(self): # TODO add kwargs and lightining capabilities
        return super().fit(self._model, self._loader)
    
