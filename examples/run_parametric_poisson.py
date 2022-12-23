import torch
from torch.nn import Softplus
from pina import Plotter, LabelTensor, PINN
from pina.model import FeedForward
from problems.parametric_poisson import ParametricPoisson

from utils import setup_generic_run_parser, setup_extra_features_parser


class myFeature(torch.nn.Module):
    """
    """
    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t = (
            torch.exp(
                - 2*(x.extract(['x']) - x.extract(['mu1']))**2
                - 2*(x.extract(['y']) - x.extract(['mu2']))**2
            )
        )
        return LabelTensor(t, ['k0'])


if __name__ == "__main__":
    # fmt: off
    args = setup_extra_features_parser(
        setup_generic_run_parser()
    ).parse_args()
    # fmt: on

    feat = [myFeature()] if args.extra else []

    poisson_problem = ParametricPoisson()
    model = FeedForward(
        layers=[10, 10, 10],
        output_variables=poisson_problem.output_variables,
        input_variables=poisson_problem.input_variables,
        func=Softplus,
        extra_features=feat
    )

    pinn = PINN(poisson_problem, model, lr=0.006, regularizer=1e-6)

    if args.save:
        pinn.span_pts(
            {'variables': ['x', 'y'], 'mode': 'random', 'n': 100},
            {'variables': ['mu1', 'mu2'], 'mode': 'grid', 'n': 5},
            locations=['D'])
        pinn.span_pts(
            {'variables': ['x', 'y'], 'mode': 'grid', 'n': 20},
            {'variables': ['mu1', 'mu2'], 'mode': 'grid', 'n': 5},
            locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])
        pinn.train(10000, 100)
        pinn.save_state(f'pina.poisson_param{args.id_run}')
    if args.load:
        pinn.load_state(f'pina.poisson_param{args.id_run}')
        plotter = Plotter()
        plotter.plot(pinn, fixed_variables={'mu1': 0, 'mu2': 1}, levels=21)
        plotter.plot(pinn, fixed_variables={'mu1': 1, 'mu2': -1}, levels=21)
