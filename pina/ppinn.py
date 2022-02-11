import torch
import numpy as np

from .problem import AbstractProblem
from . import PINN, LabelTensor
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # 3.1415927410125732


class ParametricPINN(PINN):

    def __init__(self, problem, model, optimizer=torch.optim.Adam, lr=0.001,
                 regularizer=0.00001, data_weight=1., dtype=torch.float64,
                 device='cpu', lr_accelerate=None, error_norm='mse'):
        '''
        :param Problem problem: the formualation of the problem.
        :param dict architecture: a dictionary containing the information to
            build the model. Valid options are:
            - inner_size [int] the number of neurons in the hidden layers; by
              default is 20.
            - n_layers [int] the number of hidden layers; by default is 4.
            - func [nn.Module or str] the activation function; passing a `str`
              is possible to chose adaptive function (between 'adapt_tanh'); by
              default is non-adaptive iperbolic tangent.
        :param float lr: the learning rate; default is 0.001
        :param float regularizer: the coefficient for L2 regularizer term
        :param type dtype: the data type to use for the model. Valid option are
            `torch.float32` and `torch.float64` (`torch.float16` only on GPU);
            default is `torch.float64`.
        :param float lr_accelete: the coefficient that controls the learning
            rate increase, such that, for all the epoches in which the loss is
            decreasing, the learning_rate is update using
                $learning_rate = learning_rate * lr_accelerate$.
            When the loss stops to decrease, the learning rate is set to the
            initial value [TODO test parameters]

        '''

        self.problem = problem


        # self._architecture = architecture if architecture else dict()
        # self._architecture['input_dimension'] = self.problem.domain_bound.shape[0]
        # self._architecture['output_dimension'] = len(self.problem.variables)
        # if hasattr(self.problem, 'params_domain'):
            # self._architecture['input_dimension'] += self.problem.params_domain.shape[0]

        self.accelerate = lr_accelerate

        self.error_norm = error_norm

        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError
        self.device = torch.device(device)

        self.dtype = dtype
        self.history = []

        self.model = model
        self.model.to(dtype=self.dtype, device=self.device)

        self.input_pts = {}
        self.truth_values = {}


        self.trained_epoch = 0
        self.optimizer = optimizer(
            self.model.parameters(), lr=lr, weight_decay=regularizer)

        self.data_weight = data_weight

    @property
    def problem(self):
        return self._problem

    @problem.setter
    def problem(self, problem):
        if not isinstance(problem, AbstractProblem):
            raise TypeError
        self._problem = problem

    def get_data_residuals(self):

        data_residuals = []

        for output in self.data_pts:
            data_values_pred = self.model(self.data_pts[output])
            data_residuals.append(data_values_pred - self.data_values[output])

        return torch.cat(data_residuals)

    def get_phys_residuals(self):
        """
        """

        residuals = []
        for equation in self.problem.equation:
            residuals.append(equation(self.phys_pts, self.model(self.phys_pts)))
        return residuals



    def train(self, stop=100, frequency_print=2, trial=None):

        epoch = 0

        while True:

            losses = []

            for condition_name in self.problem.conditions:
                condition = self.problem.conditions[condition_name]
                pts = self.input_pts[condition_name]

                predicted = self.model(pts.tensor)
                #predicted = self.model(pts)

                residuals = condition.function(pts, predicted)
                losses.append(self._compute_norm(residuals))
            self.optimizer.zero_grad()
            sum(losses).backward()
            self.optimizer.step()

                #for p in parameters:
                #    pts = self.input_pts[condition_name]
                #    #pts = torch.cat([pts.tensor, p.double().repeat(pts.tensor.shape[0]).reshape(-1, 2)], axis=1)
                #    #pts = torch.cat([pts.tensor, p.double().repeat(pts.tensor.shape[0]).reshape(-1, 1)], axis=1)
                #    #print(self.problem.input_variables)
                #    # print(self.problem.parameters)
                #    # print(pts.shape)
                #    print(pts.tensor.repeat_interleave(parameters.shape[0]))
                #    # print(pts)
                #    # gg
                #    a = torch.cat([
                #        pts.tensor.repeat_interleave(parameters.shape[0], dim=0),
                #        torch.tile(parameters, (pts.tensor.shape[0], 1))
                #    ], axis=1)
                #    for i in a:
                #        print(i.detach())
                #    ttt
                #    pts = LabelTensor(pts, self.problem.input_variables + self.problem.parameters)


                #    ffff
                #    print(pts.labels)

                #    predicted = self.model(pts.tensor)
                #    #predicted = self.model(pts)
                #    if isinstance(self.problem.conditions[condition_name]['func'], list):
                #        for func in self.problem.conditions[condition_name]['func']:
                #            residuals = func(pts, LabelTensor(p.reshape(1, -1), ['mu', 'alpha']), predicted)
                #            tmp_losses.append(self._compute_norm(residuals))
                #    else:
                #        residuals = self.problem.conditions[condition_name]['func'](pts, p, predicted)
                #        tmp_losses.append(self._compute_norm(residuals))
                #losses.append(sum(tmp_losses))


            self.trained_epoch += 1
            #if epoch % 10 == 0:
            #    self.history.append(losses)

            epoch += 1

            if trial:
                import optuna
                trial.report(loss[0].item()+loss[1].item(), epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if isinstance(stop, int):
                if epoch == stop:
                    break
            elif isinstance(stop, float):
                if loss[0].item() + loss[1].item() < stop:
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

    def plot(self, res, param, filename=None, variable=None):
            '''
            '''
            import matplotlib
            matplotlib.use('GTK3Agg')
            import matplotlib.pyplot as plt

            pts_container = []
            for mn, mx in [[-1, 1], [-1, 1]]:
                pts_container.append(np.linspace(mn, mx, res))
            grids_container = np.meshgrid(*pts_container)
            unrolled_pts = torch.tensor([t.flatten() for t in grids_container]).T
            unrolled_pts = torch.cat([unrolled_pts, param.double().repeat(unrolled_pts.shape[0]).reshape(-1, 2)], axis=1)

            unrolled_pts.to(dtype=self.dtype)
            unrolled_pts = LabelTensor(unrolled_pts, ['x1', 'x2', 'mu1', 'mu2'])

            Z_pred = self.model(unrolled_pts.tensor)
            n = Z_pred.tensor.shape[1]
            plt.figure(figsize=(6*n, 6))

            for i, output in enumerate(Z_pred.tensor.T, start=1):
                output = output.detach().numpy().reshape(grids_container[0].shape)
                plt.subplot(1, n, i)
                plt.contourf(*grids_container, output)
                plt.colorbar()

            if filename is None:
                plt.show()
            else:
                plt.savefig(filename)

    def plot_params(self, res, param, filename=None, variable=None):
            '''
            '''
            import matplotlib
            matplotlib.use('GTK3Agg')
            import matplotlib.pyplot as plt

            if hasattr(self.problem, 'truth_solution') and self.problem.truth_solution is not None:
                n_plot = 2
            elif hasattr(self.problem, 'data_solution') and self.problem.data_solution is not None:
                n_plot = 2
            else:
                n_plot = 1

            fig, axs = plt.subplots(nrows=1, ncols=n_plot, figsize=(n_plot*6,4))
            if not isinstance(axs, np.ndarray): axs = [axs]

            if hasattr(self.problem, 'data_solution') and self.problem.data_solution is not None:
                grids_container = self.problem.data_solution['grid']
                Z_true = self.problem.data_solution['grid_solution']
            elif hasattr(self.problem, 'truth_solution') and self.problem.truth_solution is not None:

                pts_container = []
                for mn, mx in self.problem.domain_bound:
                    pts_container.append(np.linspace(mn, mx, res))

                grids_container = np.meshgrid(*pts_container)
                Z_true = self.problem.truth_solution(*grids_container)

            pts_container = []
            for mn, mx in self.problem.domain_bound:
                pts_container.append(np.linspace(mn, mx, res))
            grids_container = np.meshgrid(*pts_container)
            unrolled_pts = torch.tensor([t.flatten() for t in grids_container]).T.to(dtype=self.type)
            #print(unrolled_pts)
            #print(param)
            param_unrolled_pts = torch.cat((unrolled_pts, param.repeat(unrolled_pts.shape[0], 1)), 1)
            if variable==None:
                variable = self.problem.variables[0]
                Z_pred = self.evaluate(param_unrolled_pts)[variable]
                variable = "Solution"
            else:
                Z_pred = self.evaluate(param_unrolled_pts)[variable]

            Z_pred= Z_pred.detach().numpy().reshape(grids_container[0].shape)
            set_pred = axs[0].contourf(*grids_container, Z_pred)
            axs[0].set_title('PINN [trained epoch = {}]'.format(self.trained_epoch) + " " + variable) #TODO add info about parameter in the title
            fig.colorbar(set_pred, ax=axs[0])

            if n_plot == 2:

                set_true = axs[1].contourf(*grids_container, Z_true)

                axs[1].set_title('Truth solution')
                fig.colorbar(set_true, ax=axs[1])

            if filename is None:
                    plt.show()
            else:
                    fig.savefig(filename + " " + variable)

    def plot_error(self, res, filename=None):
        import matplotlib
        matplotlib.use('GTK3Agg')
        import matplotlib.pyplot as plt


        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6,4))
        if not isinstance(axs, np.ndarray): axs = [axs]

        if hasattr(self.problem, 'data_solution') and self.problem.data_solution is not None:
            grids_container = self.problem.data_solution['grid']
            Z_true = self.problem.data_solution['grid_solution']
        elif hasattr(self.problem, 'truth_solution') and self.problem.truth_solution is not None:
            pts_container = []
            for mn, mx in self.problem.domain_bound:
                pts_container.append(np.linspace(mn, mx, res))

            grids_container = np.meshgrid(*pts_container)
            Z_true = self.problem.truth_solution(*grids_container)
        try:
            unrolled_pts = torch.tensor([t.flatten() for t in grids_container]).T.to(dtype=self.type)

            Z_pred = self.model(unrolled_pts)
            Z_pred = Z_pred.detach().numpy().reshape(grids_container[0].shape)
            set_pred = axs[0].contourf(*grids_container, abs(Z_pred - Z_true))
            axs[0].set_title('PINN [trained epoch = {}]'.format(self.trained_epoch) + "Pointwise Error")
            fig.colorbar(set_pred, ax=axs[0])

            if filename is None:
                    plt.show()
            else:
                    fig.savefig(filename)
        except:
            print("")
            print("Something went wrong...")
            print("Not able to plot the error. Please pass a data solution or a true solution")

'''
print(self.pred_loss.item(),loss.item(), self.old_loss.item())
if self.accelerate is not None:
    if self.pred_loss > loss and loss >= self.old_loss:
        self.current_lr = self.original_lr
        #print('restart')
    elif (loss-self.pred_loss).item() < 0.1:
        self.current_lr += .5*self.current_lr
        #print('powa')
    else:
        self.current_lr -= .5*self.current_lr
        #print(self.current_lr)
        #self.current_lr = min(loss.item()*3, 0.02)

    for g in self.optimizer.param_groups:
        g['lr'] = self.current_lr
'''
