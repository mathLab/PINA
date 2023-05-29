import argparse

from torch.nn import Softplus

from pina.model import FeedForward
from pina import Plotter, PINN
from problems.first_order_ode import FirstOrderODE


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)
    args = parser.parse_args()

    # define Problem + Model + PINN
    problem = FirstOrderODE()
    model = FeedForward(
        layers=[4]*2,
        output_variables=problem.output_variables,
        input_variables=problem.input_variables,
        func=Softplus,
    )
    pinn = PINN(problem, model, lr=0.03, error_norm='mse', regularizer=0)

    if args.s:

        pinn.span_pts(
            {'variables': ['x'], 'mode': 'grid', 'n': 1}, locations=['bc'])
        pinn.span_pts(
            {'variables': ['x'], 'mode': 'grid', 'n': 30}, locations=['dd'])
        Plotter().plot_samples(pinn, ['x'])
        pinn.train(1200, 50)
        pinn.save_state('pina.ode')

    else:
        pinn.load_state('pina.ode')
        plotter = Plotter()
        plotter.plot(pinn, components=['y'])
