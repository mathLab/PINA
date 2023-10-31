""" Solver module. """

from pytorch_lightning import Trainer
from .utils import check_consistency
from .dataset import SamplePointDataset, SamplePointLoader, DataPointDataset
from .solvers.solver import SolverInterface

class Trainer(Trainer):

    def __init__(self, solver, batch_size=None, **kwargs):
        super().__init__(**kwargs)

        # check inheritance consistency for solver
        check_consistency(solver, SolverInterface)
        self._model = solver
        self.batch_size = batch_size

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
        dataset_phys = SamplePointDataset(self._model.problem) 
        dataset_data = DataPointDataset(self._model.problem)
        self._loader = SamplePointLoader(
            dataset_phys, dataset_data, batch_size=10,
            shuffle=True)

    def train(self, **kwargs):
        return super().fit(self._model, train_dataloaders=self._loader, **kwargs)
    
    @property
    def solver(self):
        """
        Returning trainer solver.
        """
        return self._model