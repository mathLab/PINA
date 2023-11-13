import torch
from lightning.pytorch.callbacks import Callback
from pina import LabelTensor

class StochasticRefinement(Callback):

    def __init__(self, sample_every, eps=0.1):
        super().__init__()
        self._sample_every = sample_every
        self._eps = eps
            
    def _compute_residual(self, trainer):
        # extract the solver and device from trainer
        solver = trainer._model
        device = trainer._accelerator_connector._accelerator_flag

        # compute residual
        res_loss = {}
        tot_loss = []
        for location in self._sampling_locations:
            condition = solver.problem.conditions[location]
            pts = solver.problem.input_pts[location]
            # send points to correct device
            pts = pts.to(device)
            pts = pts.requires_grad_(True)
            pts.retain_grad()
            # PINN loss: equation evaluated only on locations where sampling is needed
            target = condition.equation.residual(pts, solver.forward(pts))
            res_loss[location] = torch.abs(target)
            tot_loss.append(torch.abs(target))
    
        return res_loss

    def _eps_greedy_sampling(self, pts, residuals):
        new_pts = []
        old_pts_idx = torch.argsort(residuals, descending=True)  # select points with higher residuals
        old_pts = pts[old_pts_idx]
        old_pts_shape = old_pts[0].shape
        count = 0

        for _ in range(pts.shape[0]):
            rd_pt = torch.rand(1)
            if rd_pt < self._eps:
                pt = 2 * torch.rand(size=old_pts_shape) -1
                new_pts.append(pt)

            else:
                new_pts.append(old_pts[count])
                count+=1

        return torch.vstack(new_pts)

    def _routine(self, trainer):

        # compute residual (all device possible)
        res_loss = self._compute_residual(trainer)

        # !!!!!! From now everything is performed on CPU !!!!!!

        tot_points = 0
        for location in self._sampling_locations:
            pts = trainer._model.problem.input_pts[location]
            labels = pts.labels
            pts = pts.cpu().detach()
            residuals = res_loss[location].cpu()
            pts = self._eps_greedy_sampling(pts, residuals)
            pts = LabelTensor(pts, labels)
            tot_points += len(pts)
            trainer._model.problem.input_pts[location] = pts

        # update dataloader
        trainer._create_or_update_loader()

    def on_train_start(self, trainer, _):
        # extract locations for sampling
        problem = trainer._model.problem
        locations = []
        for condition_name in problem.conditions:
            condition = problem.conditions[condition_name]
            if hasattr(condition, 'location'):
                locations.append(condition_name)
        self._sampling_locations = locations

    def on_train_epoch_end(self, trainer, __):
        if trainer.current_epoch % self._sample_every == 0:
            self._routine(trainer)


class AdaptiveSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.sigm = torch.nn.Sigmoid()
        self.scale = torch.nn.Parameter(torch.rand(1))
        self.scale.requiresGrad = True

    def forward(self, x):
        return self.sigm(self.scale*x)
    
class AdaptiveSwish(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.sigm = torch.nn.Sigmoid()
        self.scale = torch.nn.Parameter(torch.rand(1))
        self.scale.requiresGrad = True

    def forward(self, x):
        return x * self.sigm(self.scale*x)
