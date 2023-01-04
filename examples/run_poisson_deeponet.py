import argparse
import logging

import torch
from problems.poisson import Poisson

from pina import PINN, LabelTensor, Plotter
from pina.model.deeponet import DeepONet, check_combos, spawn_combo_networks

logging.basicConfig(
    filename="poisson_deeponet.log", filemode="w", level=logging.INFO
)


class SinFeature(torch.nn.Module):
    """
    Feature: sin(x)
    """

    def __init__(self, label):
        super().__init__()

        if not isinstance(label, (tuple, list)):
            label = [label]
        self._label = label

    def forward(self, x):
        """
        Defines the computation performed at every call.

        :param LabelTensor x: the input tensor.
        :return: the output computed by the model.
        :rtype: LabelTensor
        """
        t = torch.sin(x.extract(self._label) * torch.pi)
        return LabelTensor(t, [f"sin({self._label})"])


def prepare_deeponet_model(args, problem, extra_feature_combo_func=None):
    combos = tuple(map(lambda combo: combo.split("-"), args.combos.split(",")))
    check_combos(combos, problem.input_variables)

    extra_feature = extra_feature_combo_func if args.extra else None
    networks = spawn_combo_networks(
        combos=combos,
        layers=list(map(int, args.layers.split(","))) if args.layers else [],
        output_dimension=args.hidden * len(problem.output_variables),
        func=torch.nn.Softplus,
        extra_feature=extra_feature,
        bias=not args.nobias,
    )

    return DeepONet(
        networks,
        problem.output_variables,
        aggregator=args.aggregator,
        reduction=args.reduction,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PINA")
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-l", "--load", action="store_true")
    parser.add_argument("id_run", help="Run ID", type=int)

    parser.add_argument("--extra", help="Extra features", action="store_true")
    parser.add_argument("--nobias", action="store_true")
    parser.add_argument(
        "--combos",
        help="DeepONet internal network combinations",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--aggregator", help="Aggregator for DeepONet", type=str, default="*"
    )
    parser.add_argument(
        "--reduction", help="Reduction for DeepONet", type=str, default="+"
    )
    parser.add_argument(
        "--hidden",
        help="Number of variables in the hidden DeepONet layer",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--layers",
        help="Structure of the DeepONet partial layers",
        type=str,
        required=True,
    )
    cli_args = parser.parse_args()

    poisson_problem = Poisson()

    model = prepare_deeponet_model(
        cli_args,
        poisson_problem,
        extra_feature_combo_func=lambda combo: [SinFeature(combo)],
    )
    pinn = PINN(poisson_problem, model, lr=0.01, regularizer=1e-8)
    if cli_args.save:
        pinn.span_pts(
            20, "grid", locations=["gamma1", "gamma2", "gamma3", "gamma4"]
        )
        pinn.span_pts(20, "grid", locations=["D"])
        pinn.train(1.0e-10, 100)
        pinn.save_state(f"pina.poisson_{cli_args.id_run}")
    if cli_args.load:
        pinn.load_state(f"pina.poisson_{cli_args.id_run}")
        plotter = Plotter()
        plotter.plot(pinn)
