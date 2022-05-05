from .problem import AbstractProblem
import torch
import matplotlib.pyplot as plt
import numpy as np
from pina.label_tensor import LabelTensor
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732


class PINN(object):

    def __init__(self,
            problem,
            model,
            optimizer=torch.optim.Adam,
            lr=0.001,
            regularizer=0.00001,
            dtype=torch.float32,
            device='cpu',
            error_norm='mse'):
        '''
        :param Problem problem: the formualation of the problem.
        :param torch.nn.Module model: the neural network model to use.
        :param float lr: the learning rate; default is 0.001.
        :param float regularizer: the coefficient for L2 regularizer term.
        :param type dtype: the data type to use for the model. Valid option are
            `torch.float32` and `torch.float64` (`torch.float16` only on GPU);
            default is `torch.float64`.
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
        self.history = []

        self.model = model
        self.model.to(dtype=self.dtype, device=self.device)

        self.truth_values = {}
        self.input_pts = {}

        self.trained_epoch = 0
        self.optimizer = optimizer(
            self.model.parameters(), lr=lr, weight_decay=regularizer)

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
            return torch.sum(torch.abs(vec**self.error_norm))**(1./self.error_norm)
        elif self.error_norm == 'mse':
            return torch.mean(vec**2)
        elif self.error_norm == 'me':
            return torch.mean(torch.abs(vec))
        else:
            raise RuntimeError

    def save_state(self, filename):

        checkpoint = {
                'epoch': self.trained_epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state' : self.optimizer.state_dict(),
                'optimizer_class' : self.optimizer.__class__,
                'history' : self.history,
                'input_points_dict' : self.input_pts,
        }

        # TODO save also architecture param?
        #if isinstance(self.model, DeepFeedForward):
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
        self.history = checkpoint['history']

        self.input_pts = checkpoint['input_points_dict']

        return self


    def span_pts(self, n_spatial, n_params=0, mode_spatial='grid', mode_param='random', locations='all'):
        if locations == 'all':
            locations = [condition for condition in self.problem.conditions]
        for location in locations:
            condition = self.problem.conditions[location]

            try:
                pts = condition.location.sample(n_spatial, mode_spatial, variables=self.problem.spatial_variables)
                if n_params != 0:
                    pts_params = condition.location.sample(n_params, mode_param, variables=self.problem.parameters)
                    pts = LabelTensor(pts.repeat(n_params, 1), pts.labels)
                    pts_params = LabelTensor(pts_params.repeat_interleave(n_spatial).reshape((n_spatial*n_params, len(self.problem.parameters))), pts_params.labels)
                    pts = pts.append(pts_params)
            except:
                pts = condition.input_points
            self.input_pts[location] = pts  #.double()  # TODO
            self.input_pts[location] = (
                self.input_pts[location].to(dtype=self.dtype,
                                            device=self.device))
            self.input_pts[location].requires_grad_(True)
            self.input_pts[location].retain_grad()


    def plot_pts(self, locations='all'):
        import matplotlib
       # matplotlib.use('GTK3Agg')
        if locations == 'all':
            locations = [condition for condition in self.problem.conditions]
        for location in locations:
            x = self.input_pts[location].extract(['x'])
            y = self.input_pts[location].extract(['y'])
            plt.plot(x.detach(), y.detach(), '.', label=location)
      #      np.savetxt('burgers_{}_pts.txt'.format(location), self.input_pts[location].tensor.detach(), header='x y', delimiter=' ')
        plt.legend()
        plt.show()



    def train(self, stop=100, frequency_print=2, trial=None):

        epoch = 0

        while True:

            losses = []

            for condition_name in self.problem.conditions:
                condition = self.problem.conditions[condition_name]
                pts = self.input_pts[condition_name]
                predicted = self.model(pts)
                if hasattr(condition, 'function'):
                    if isinstance(condition.function, list):
                        for function in condition.function:
                            residuals = function(pts, predicted)
                            local_loss = condition.data_weight*self._compute_norm(residuals)
                            losses.append(local_loss)
                    else:
                        residuals = condition.function(pts, predicted)
                        local_loss = condition.data_weight*self._compute_norm(residuals)
                        losses.append(local_loss)
                elif hasattr(condition, 'output_points'):
                    residuals = predicted - condition.output_points
                    local_loss = condition.data_weight*self._compute_norm(residuals)
                    losses.append(local_loss)
            self.optimizer.zero_grad()

            sum(losses).backward()
            self.optimizer.step()

            self.trained_epoch += 1
            if epoch % 50 == 0:
                self.history.append([loss.detach().item() for loss in losses])
            epoch += 1

            if trial:
                import optuna
                trial.report(sum(losses), epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if isinstance(stop, int):
                if epoch == stop:
                    break
            elif isinstance(stop, float):
                if sum(losses) < stop:
                    break

            if epoch % frequency_print == 0:
                print('[epoch {:05d}] {:.6e} '.format(self.trained_epoch, sum(losses).item()), end='')
                for loss in losses:
                    print('{:.6e} '.format(loss), end='')
                print()

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
            unrolled_pts = torch.tensor([t.flatten() for t in grids_container]).T.to(dtype=self.dtype, device=self.device)
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
            print("Not able to compute the error. Please pass a data solution or a true solution")
