import torch

from pina import Trainer, LabelTensor
from pina.plotter import Plotter
from pina.solvers import PINN
from pina.model import FeedForward, ResidualFeedForward, DeepONet, MultiFeedForward
from pina.callbacks import SwitchOptimizer, R3Refinement
from multiphase_problems import ZalesakDisck
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
    
# make the problem and sample points
problem = ZalesakDisck()
problem.discretise_domain(500, 'random', locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])
problem.discretise_domain(20, 'grid', locations=['t0'])
problem.discretise_domain(20, 'grid', locations=['D'], variables=['x','y'])
problem.discretise_domain(50, 'grid', locations=['D'], variables=['t'])


class MyRelu(torch.nn.Hardtanh):

    def __init__(self) -> None:
        super().__init__(0., 2*torch.pi)
    

class Rotation(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.alpha = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False),
                                         MyRelu())
        self.w = torch.nn.Parameter(torch.tensor([0.]))
        self.f =  MyRelu()
                                   
    def _create_rot_matrix(self, angle):
        # Create rotation matrices for each angle
        cosine = torch.cos(angle)
        sine = torch.sin(angle)

        # Create the rotation matrices
        rotation_matrix = torch.cat([cosine, -sine, sine, cosine], dim=1)
        rotation_matrix = rotation_matrix.view(-1, 2, 2)

        return rotation_matrix
    
    def forward(self, x):
        angle = self.f(x.extract(['t']) * self.w ).as_subclass(torch.Tensor)

        #angle = self.alpha(x.extract(['t'])).as_subclass(torch.Tensor)
        x_not_rotated = x.extract(['x', 'y']).as_subclass(torch.Tensor).unsqueeze(2)

        # rot matrix
        rotation_matrix = self._create_rot_matrix(angle) # N x 2 x 2

        x_rotated = torch.bmm(rotation_matrix, x_not_rotated).squeeze(2).as_subclass(LabelTensor)
        x_rotated.labels = ['x', 'y']

        return ZalesakDisck.phi_initial(x_rotated.extract(['x', 'y']))
    
    
class ExtraFeat(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, x):
        phi = ZalesakDisck.phi_initial(x.extract(['x', 'y']))
        return LabelTensor(phi, ['phi0'])


model = Rotation()

# make the solver
solver = PINN(problem=problem,
              model=model, 
              optimizer=torch.optim.Adam,
              extra_features=[ExtraFeat()],
              optimizer_kwargs={'lr' : 0.001})

# make the trainer
trainer = Trainer(solver=solver, max_epochs=2000)
trainer.train()


# plotter
pl = Plotter()
times = torch.linspace(0, 4, 10)
for i, t in enumerate(times):
    pl.plot(solver, components=['x', 'y'], fixed_variables={'t':t})
    #pl.plot(trainer=trainer, components=['x', 'y'], fixed_variables={'t':t}, filename=f'images/image_{i}.png')