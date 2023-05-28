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
        mydata = solver.problem.input_pts
        # TODO
        # this works only for sampled points
        # we need to add the capability to get input/output points
        # from problem.
        self._loader = DummyLoader(mydata) 


    def train(self): # TODO add kwargs and lightining capabilities
        return super().fit(self._model, self._loader)
    
