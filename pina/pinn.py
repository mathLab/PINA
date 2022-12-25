""" Module for PINN """
import torch

from .problem import AbstractProblem
from .label_tensor import LabelTensor
from .utils import merge_tensors, PinaDataset


torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732


class PINN(object):

    def __init__(self,
                 problem,
                 model,
                 optimizer=torch.optim.Adam,
                 lr=0.001,
                 regularizer=0.00001,
                 batch_size=None,
                 dtype=torch.float32,
                 device='cpu',
                 error_norm='mse'):
        '''
        :param Problem problem: the formualation of the problem.
        :param torch.nn.Module model: the neural network model to use.
        :param torch.optim optimizer: the neural network optimizer to use;
            default is `torch.optim.Adam`.
        :param float lr: the learning rate; default is 0.001.
        :param float regularizer: the coefficient for L2 regularizer term.
        :param type dtype: the data type to use for the model. Valid option are
            `torch.float32` and `torch.float64` (`torch.float16` only on GPU);
            default is `torch.float64`.
        :param string device: the device used for training; default 'cpu'
            option include 'cuda' if cuda is available.
        :param string/int error_norm: the loss function used as minimizer,
            default mean square error 'mse'. If string options include mean
            error 'me' and mean square error 'mse'. If int, the p-norm is
            calculated where p is specifined by the int input.
        :param int batch_size: batch size for the dataloader; default 5.
        '''

        if dtype == torch.float64:
            raise NotImplementedError('only float for now')

        self.problem = problem

        # self._architecture = architecture if architecture else dict()
        # self._architecture['input_dimension'] = self.problem.domain_bound.shape[0]
        # self._architecture['output_dimension'] = len(self.problem.variables)
        # if hasattr(self.problem, 'params_domain'):
        # self._architecture['input_dimension'] += self.problem.params_domain.shape[0]

        self.error_norm = error_norm

        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError
        self.device = torch.device(device)

        self.dtype = dtype
        self.history_loss = {}

        self.model = model
        self.model.to(dtype=self.dtype, device=self.device)

        self.truth_values = {}
        self.input_pts = {}

        self.trained_epoch = 0
        self.optimizer = optimizer(
            self.model.parameters(), lr=lr, weight_decay=regularizer)

        self.batch_size = batch_size
        self.data_set = PinaDataset(self)

    @property
    def problem(self):
        return self._problem

    @problem.setter
    def problem(self, problem):
        if not isinstance(problem, AbstractProblem):
            raise TypeError
        self._problem = problem

    def _compute_norm(self, vec):
        """
        Compute the norm of the `vec` one-dimensional tensor based on the
        `self.error_norm` attribute.

        .. todo: complete

        :param vec torch.tensor: the tensor
        """
        if isinstance(self.error_norm, int):
            return torch.linalg.vector_norm(vec, ord=self.error_norm,  dtype=self.dytpe)
        elif self.error_norm == 'mse':
            return torch.mean(vec.pow(2))
        elif self.error_norm == 'me':
            return torch.mean(torch.abs(vec))
        else:
            raise RuntimeError

    def save_state(self, filename):

        checkpoint = {
            'epoch': self.trained_epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_class': self.optimizer.__class__,
            'history': self.history_loss,
            'input_points_dict': self.input_pts,
        }

        # TODO save also architecture param?
        # if isinstance(self.model, DeepFeedForward):
        #    checkpoint['model_class'] = self.model.__class__
        #    checkpoint['model_structure'] = {
        #    }
        torch.save(checkpoint, filename)

    def load_state(self, filename):

        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state'])

        self.optimizer = checkpoint['optimizer_class'](self.model.parameters())
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        self.trained_epoch = checkpoint['epoch']
        self.history_loss = checkpoint['history']

        self.input_pts = checkpoint['input_points_dict']

        return self

    def _create_dataloader(self):
        """Private method for creating dataloader

        :return: dataloader
        :rtype: torch.utils.data.DataLoader
        """
        if self.batch_size is None:
            return [self.input_pts]

        def custom_collate(batch):
            # extracting pts labels
            _, pts = list(batch[0].items())[0]
            labels = pts.labels
            # calling default torch collate
            collate_res = default_collate(batch)
            # save collate result in dict
            res = {}
            for key, val in collate_res.items():
                val.labels = labels
                res[key] = val
            return res

        # creating dataset, list of dataset for each location
        datasets = [MyDataSet(key, val)
                    for key, val in self.input_pts.items()]
        # creating dataloader
        dataloaders = [DataLoader(dataset=dat,
                                  batch_size=self.batch_size,
                                  collate_fn=custom_collate)
                       for dat in datasets]

        return dict(zip(self.input_pts.keys(), dataloaders))

    def span_pts(self, *args, **kwargs):
        """
        >>> pinn.span_pts(n=10, mode='grid')
        >>> pinn.span_pts(n=10, mode='grid', location=['bound1'])
        >>> pinn.span_pts(n=10, mode='grid', variables=['x'])
        """

        if all(key in kwargs for key in ['n', 'mode']):
            argument = {}
            argument['n'] = kwargs['n']
            argument['mode'] = kwargs['mode']
            argument['variables'] = self.problem.input_variables
            arguments = [argument]
        elif any(key in kwargs for key in ['n', 'mode']) and args:
            raise ValueError("Don't mix args and kwargs")
        elif isinstance(args[0], int) and isinstance(args[1], str):
            argument = {}
            argument['n'] = int(args[0])
            argument['mode'] = args[1]
            argument['variables'] = self.problem.input_variables
            arguments = [argument]
        elif all(isinstance(arg, dict) for arg in args):
            arguments = args
        else:
            raise RuntimeError

        locations = kwargs.get('locations', 'all')

        if locations == 'all':
            locations = [condition for condition in self.problem.conditions]
        for location in locations:
            condition = self.problem.conditions[location]

            samples = tuple(condition.location.sample(
                            argument['n'],
                            argument['mode'],
                            variables=argument['variables'])
                            for argument in arguments)
            pts = merge_tensors(samples)

            # TODO
            # pts = pts.double()
            self.input_pts[location] = pts

    def train(self, stop=100, frequency_print=2, save_loss=1, trial=None):

        self.model.train()
        epoch = 0
        data_loader = self.data_set.dataloader

        header = []
        for condition_name in self.problem.conditions:
            condition = self.problem.conditions[condition_name]

            if hasattr(condition, 'function'):
                if isinstance(condition.function, list):
                    for function in condition.function:
                        header.append(f'{condition_name}{function.__name__}')

                    continue

            header.append(f'{condition_name}')

        while True:

            losses = []

            for condition_name in self.problem.conditions:
                condition = self.problem.conditions[condition_name]

                for batch in data_loader[condition_name]:

                    single_loss = []

                    if hasattr(condition, 'function'):
                        pts = batch[condition_name]
                        pts = pts.to(dtype=self.dtype, device=self.device)
                        pts.requires_grad_(True)
                        pts.retain_grad()

                        predicted = self.model(pts)
                        for function in condition.function:
                            residuals = function(pts, predicted)
                            local_loss = (
                                condition.data_weight*self._compute_norm(
                                    residuals))
                            single_loss.append(local_loss)
                    elif hasattr(condition, 'output_points'):
                        pts = condition.input_points.to(
                            dtype=self.dtype, device=self.device)
                        predicted = self.model(pts)
                        residuals = predicted - condition.output_points
                        local_loss = (
                            condition.data_weight*self._compute_norm(residuals))
                        single_loss.append(local_loss)

                    self.optimizer.zero_grad()
                    sum(single_loss).backward()
                    self.optimizer.step()

                losses.append(sum(single_loss))

            if save_loss and (epoch % save_loss == 0 or epoch == 0):
                self.history_loss[epoch] = [
                    loss.detach().item() for loss in losses]

            if trial:
                import optuna
                trial.report(sum(losses), epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if isinstance(stop, int):
                if epoch == stop:
                    print('[epoch {:05d}] {:.6e} '.format(
                        self.trained_epoch, sum(losses).item()), end='')
                    for loss in losses:
                        print('{:.6e} '.format(loss.item()), end='')
                    print()
                    break
            elif isinstance(stop, float):
                if sum(losses) < stop:
                    break

            if epoch % frequency_print == 0 or epoch == 1:
                print('       {:5s}  {:12s} '.format('', 'sum'),  end='')
                for name in header:
                    print('{:12.12s} '.format(name), end='')
                print()

                print('[epoch {:05d}] {:.6e} '.format(
                    self.trained_epoch, sum(losses).item()), end='')
                for loss in losses:
                    print('{:.6e} '.format(loss.item()), end='')
                print()

            self.trained_epoch += 1
            epoch += 1

        self.model.eval()

        return sum(losses).item()

    def error(self, dtype='l2', res=100):

        import numpy as np
        if hasattr(self.problem, 'truth_solution') and self.problem.truth_solution is not None:
            pts_container = []
            for mn, mx in self.problem.domain_bound:
                pts_container.append(np.linspace(mn, mx, res))
            grids_container = np.meshgrid(*pts_container)
            Z_true = self.problem.truth_solution(*grids_container)

        elif hasattr(self.problem, 'data_solution') and self.problem.data_solution is not None:
            grids_container = self.problem.data_solution['grid']
            Z_true = self.problem.data_solution['grid_solution']
        try:
            unrolled_pts = torch.tensor([t.flatten() for t in grids_container]).T.to(
                dtype=self.dtype, device=self.device)
            Z_pred = self.model(unrolled_pts)
            Z_pred = Z_pred.detach().numpy().reshape(grids_container[0].shape)

            if dtype == 'l2':
                return np.linalg.norm(Z_pred - Z_true)/np.linalg.norm(Z_true)
            else:
                # TODO H1
                pass
        except:
            print("")
            print("Something went wrong...")
            print(
                "Not able to compute the error. Please pass a data solution or a true solution")
