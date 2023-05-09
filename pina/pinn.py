""" Module for PINN """
import torch
import torch.optim.lr_scheduler as lrs

from .problem import AbstractProblem
from .model import Network
from .label_tensor import LabelTensor
from .utils import merge_tensors, PinaDataset


torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732


class PINN(object):

    def __init__(self,
                 problem,
                 model,
                 extra_features=None,
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs=None,
                 lr=0.001,
                 lr_scheduler_type=lrs.ConstantLR,
                 lr_scheduler_kwargs={"factor": 1, "total_iters": 0},
                 regularizer=0.00001,
                 batch_size=None,
                 dtype=torch.float32,
                 device='cpu',
                 error_norm='mse'):
        '''
        :param AbstractProblem problem: the formualation of the problem.
        :param torch.nn.Module model: the neural network model to use.
        :param torch.nn.Module extra_features: the additional input
            features to use as augmented input.
        :param torch.optim.Optimizer optimizer: the neural network optimizer to
            use; default is `torch.optim.Adam`.
        :param dict optimizer_kwargs: Optimizer constructor keyword args.
        :param float lr: the learning rate; default is 0.001.
        :param torch.optim.LRScheduler lr_scheduler_type: Learning
            rate scheduler.
        :param dict lr_scheduler_kwargs: LR scheduler constructor keyword args.
        :param float regularizer: the coefficient for L2 regularizer term.
        :param type dtype: the data type to use for the model. Valid option are
            `torch.float32` and `torch.float64` (`torch.float16` only on GPU);
            default is `torch.float64`.
        :param str device: the device used for training; default 'cpu'
            option include 'cuda' if cuda is available.
        :param (str, int) error_norm: the loss function used as minimizer,
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


        self.model = Network(model=model,
                             input_variables=problem.input_variables,
                             output_variables=problem.output_variables,
                             extra_features=extra_features)

        self.model.to(dtype=self.dtype, device=self.device)

        self.truth_values = {}
        self.input_pts = {}

        self.trained_epoch = 0

        if not optimizer_kwargs:
            optimizer_kwargs = {}
        optimizer_kwargs['lr'] = lr
        self.optimizer = optimizer(
            self.model.parameters(), weight_decay=regularizer, **optimizer_kwargs)
        self._lr_scheduler = lr_scheduler_type(
            self.optimizer, **lr_scheduler_kwargs)

        self.batch_size = batch_size
        self.data_set = PinaDataset(self)

    @property
    def problem(self):
        """ The problem formulation."""
        return self._problem

    @problem.setter
    def problem(self, problem):
        """
        Set the problem formulation."""
        if not isinstance(problem, AbstractProblem):
            raise TypeError
        self._problem = problem

    def _compute_norm(self, vec):
        """
        Compute the norm of the `vec` one-dimensional tensor based on the
        `self.error_norm` attribute.     

        .. todo: complete

        :param torch.Tensor vec: the tensor
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
        """
        Save the state of the model.

        :param str filename: the filename to save the state to.
        """
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
        """
        Load the state of the model.
        
        :param str filename: the filename to load the state from.
        """

        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state'])

        self.optimizer = checkpoint['optimizer_class'](self.model.parameters())
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        self.trained_epoch = checkpoint['epoch']
        self.history_loss = checkpoint['history']

        self.input_pts = checkpoint['input_points_dict']

        return self

    def span_pts(self, *args, **kwargs):
        """
        Generate a set of points to span the `Location` of all the conditions of
        the problem.

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

    def _residual_loss(self, input_pts, equation):
        """
        Compute the residual loss for a given condition.

        :param torch.Tensor pts: the points to evaluate the residual at.
        :param Equation equation: the equation to evaluate the residual with.
        """

        input_pts = input_pts.to(dtype=self.dtype, device=self.device)
        input_pts.requires_grad_(True)
        input_pts.retain_grad()

        predicted = self.model(input_pts)
        residuals = equation.residual(input_pts, predicted)
        return self._compute_norm(residuals)

    def _data_loss(self, input_pts, output_pts):
        """
        Compute the residual loss for a given condition.

        :param torch.Tensor pts: the points to evaluate the residual at.
        :param Equation equation: the equation to evaluate the residual with.
        """
        input_pts = input_pts.to(dtype=self.dtype, device=self.device)
        output_pts = output_pts.to(dtype=self.dtype, device=self.device)
        predicted = self.model(pts)
        residuals = predicted - output_pts
        return self._compute_norm(residuals)
 

    def closure(self):
        """
        """
        self.optimizer.zero_grad()

        condition_losses = []
        from torch.utils.data import DataLoader
        from .utils import MyDataset
        loader = DataLoader(
            MyDataset(self.input_pts),
            batch_size=self.batch_size,
            num_workers=1
        )
        for condition_name in self.problem.conditions:
            condition = self.problem.conditions[condition_name]

            batch_losses = []
            for batch in data_loader[condition_name]:

                if hasattr(condition, 'equation'):
                    loss = self._residual_loss(
                        batch[condition_name], condition.equation)
                elif hasattr(condition, 'output_points'):
                    loss = self._data_loss(
                        batch[condition_name], condition.output_points)

                batch_losses.append(loss * condition.data_weight)

            condition_losses.append(sum(batch_losses))

        loss = sum(condition_losses)
        loss.backward()
        return loss

    def closure(self):
        """
        """
        self.optimizer.zero_grad()

        losses = []
        for i, batch in enumerate(self.loader):

            condition_losses = []

            for condition_name, samples in batch.items():

                if condition_name not in self.problem.conditions:
                    raise RuntimeError('Something wrong happened.')

                if samples.nelement() == 0:
                    continue

                condition = self.problem.conditions[condition_name]
                print(samples)

                if hasattr(condition, 'equation'):
                    loss = self._residual_loss(samples, condition.equation)
                elif hasattr(condition, 'output_points'):
                    loss = self._data_loss(samples, condition.output_points)

                condition_losses.append(loss * condition.data_weight)

            losses.append(sum(condition_losses))

        loss = sum(losses)
        loss.backward()
        return loss


        # for condition_name in self.problem.conditions:
        #     condition = self.problem.conditions[condition_name]

        #     batch_losses = []

        #         if hasattr(condition, 'equation'):
        #             loss = self._residual_loss(
        #                 batch[condition_name], condition.equation)
        #         elif hasattr(condition, 'output_points'):
        #             loss = self._data_loss(
        #                 batch[condition_name], condition.output_points)

        #         batch_losses.append(loss * condition.data_weight)

        #     condition_losses.append(sum(batch_losses))

        # loss = sum(condition_losses)
        # loss.backward()
        # return loss

    def train(self, stop=100, frequency_print=2, save_loss=1, trial=None):

        # from .utils import MyDataset
        # print(self.input_pts)
        # ttttt
        self.model.train()
        epoch = 0
        # Add all condition with `input_points` to dataloader
        for condition in list(set(self.problem.conditions.keys()) - set(self.input_pts.keys())):
            self.input_pts[condition] = self.problem.conditions[condition]

        #data_loader = self.data_set.dataloader

        from torch.utils.data import DataLoader
        from .utils import MyDataset
        self.loader = DataLoader(
            MyDataset(self.input_pts),
            batch_size=self.batch_size,
            num_workers=1
        )
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


            # condition_losses = []
            # for condition_name in self.problem.conditions:
            #     condition = self.problem.conditions[condition_name]

            #     batch_losses = []
            #     for batch in data_loader[condition_name]:

            #         if hasattr(condition, 'equation'):
            #             loss = self._residual_loss(
            #                 batch[condition_name], condition.equation)
            #         elif hasattr(condition, 'output_points'):
            #             loss = self._data_loss(
            #                 batch[condition_name], condition.output_points)

            #         batch_losses.append(loss * condition.data_weight)

            #     condition_losses.append(sum(batch_losses))

            # self.optimizer.zero_grad()
            # sum(condition_losses).backward()
            # self.optimizer.step()
            self.optimizer.step(closure=self.closure)


            self._lr_scheduler.step()

            if save_loss and (epoch % save_loss == 0 or epoch == 0):
                self.history_loss[epoch] = [
                    loss.detach().item() for loss in condition_losses]

            if trial:
                import optuna
                trial.report(sum(losses), epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if isinstance(stop, int):
                if epoch == stop:
                    print('[epoch {:05d}] {:.6e} '.format(
                        self.trained_epoch, sum(condition_losses).item()), end='')
                    for loss in condition_losses:
                        print('{:.6e} '.format(loss.item()), end='')
                    print()
                    break
            elif isinstance(stop, float):
                if sum(condition_losses) < stop:
                    break

            if epoch % frequency_print == 0 or epoch == 1:
                print('       {:5s}  {:12s} '.format('', 'sum'),  end='')
                for name in header:
                    print('{:12.12s} '.format(name), end='')
                print()

                print('[epoch {:05d}] {:.6e} '.format(
                    self.trained_epoch, sum(condition_losses).item()), end='')
                for loss in condition_losses:
                    print('{:.6e} '.format(loss.item()), end='')
                print()

            self.trained_epoch += 1
            epoch += 1

        self.model.eval()

        return sum(condition_losses).item()

    # def error(self, dtype='l2', res=100):

    #     import numpy as np
    #     if hasattr(self.problem, 'truth_solution') and self.problem.truth_solution is not None:
    #         pts_container = []
    #         for mn, mx in self.problem.domain_bound:
    #             pts_container.append(np.linspace(mn, mx, res))
    #         grids_container = np.meshgrid(*pts_container)
    #         Z_true = self.problem.truth_solution(*grids_container)

    #     elif hasattr(self.problem, 'data_solution') and self.problem.data_solution is not None:
    #         grids_container = self.problem.data_solution['grid']
    #         Z_true = self.problem.data_solution['grid_solution']
    #     try:
    #         unrolled_pts = torch.tensor([t.flatten() for t in grids_container]).T.to(
    #             dtype=self.dtype, device=self.device)
    #         Z_pred = self.model(unrolled_pts)
    #         Z_pred = Z_pred.detach().numpy().reshape(grids_container[0].shape)

    #         if dtype == 'l2':
    #             return np.linalg.norm(Z_pred - Z_true)/np.linalg.norm(Z_true)
    #         else:
    #             # TODO H1
    #             pass
    #     except:
    #         print("")
    #         print("Something went wrong...")
    #         print(
    #             "Not able to compute the error. Please pass a data solution or a true solution")
