import argparse
import logging

import torch
from problems.poisson import Poisson

from pina import PINN, LabelTensor, Plotter
from pina.model import DeepONet, FeedForward


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


class myRBF(torch.nn.Module):
    def __init__(self, input_):

        super().__init__()

        self.input_variables = [input_]
        self.a = torch.nn.Parameter(torch.tensor([-.3]))
        # self.b = torch.nn.Parameter(torch.tensor([0.5]))
        self.b = torch.tensor([0.5])
        self.c = torch.nn.Parameter(torch.tensor([.5]))

    def forward(self, x):
        x = x.extract(self.input_variables)
        result = self.a * torch.exp(-(x - self.b)**2/(self.c**2))
        return result

class myModel(torch.nn.Module):
    def __init__(self):

        super().__init__()
        self.ffn_x = myRBF('x')
        self.ffn_y = myRBF('y')

    def forward(self, x):
        result =  self.ffn_x(x) * self.ffn_y(x)
        result.labels = ['u']
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PINA")
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-l", "--load", action="store_true")
    parser.add_argument("id_run", help="Run ID", type=int)

    parser.add_argument("--extra", help="Extra features", action="store_true")
    args = parser.parse_args()

    problem = Poisson()

    # ffn_x = FeedForward(
    #     input_variables=['x'], layers=[], output_variables=1,
    #     func=torch.nn.Softplus,
    #     extra_features=[SinFeature('x')]
    # )
    # ffn_y = FeedForward
    #     input_variables=['y'], layers=[], output_variables=1,
    #     func=torch.nn.Softplus,
    #     extra_features=[SinFeature('y')]
    # )
    model = myModel()
    test = torch.tensor([[0.0, 0.5]])
    test.labels = ['x', 'y']
    pinn = PINN(problem, model, lr=0.0001)

    if args.save:
        pinn.span_pts(
            20, "grid", locations=["gamma1", "gamma2", "gamma3", "gamma4"]
        )
        pinn.span_pts(20, "grid", locations=["D"])
        while True:
            pinn.train(500, 50)
            print(model.ffn_x.a)
            print(model.ffn_x.b)
            print(model.ffn_x.c)

            xi = torch.linspace(0, 1, 64).reshape(-1, 1).as_subclass(LabelTensor)
            xi.labels = ['x']
            yi = model.ffn_x(xi)
            y_truth = -torch.sin(xi*torch.pi)

            import matplotlib.pyplot as plt
            plt.plot(xi.detach().flatten(), yi.detach().flatten(), 'r-')
            plt.plot(xi.detach().flatten(), y_truth.detach().flatten(), 'b-')
            plt.plot(xi.detach().flatten(), -y_truth.detach().flatten(), 'b-')
            plt.show()
        pinn.save_state(f"pina.poisson_{args.id_run}")

    if args.load:
        pinn.load_state(f"pina.poisson_{args.id_run}")
        plotter = Plotter()
        plotter.plot(pinn)
