import torch
import pytest
from pina import Condition, LabelTensor, Graph
from pina.condition import InputOutputPointsCondition, DomainEquationCondition
from pina.graph import RadiusGraph
from pina.problem import AbstractProblem, SpatialProblem
from pina.domain import CartesianDomain
from pina.equation.equation import Equation
from pina.equation.equation_factory import FixedValue
from pina.operator import laplacian
from pina.collector import Collector


def test_supervised_tensor_collector():
    class SupervisedProblem(AbstractProblem):
        output_variables = None
        conditions = {
            "data1": Condition(
                input_points=torch.rand((10, 2)),
                output_points=torch.rand((10, 2)),
            ),
            "data2": Condition(
                input_points=torch.rand((20, 2)),
                output_points=torch.rand((20, 2)),
            ),
            "data3": Condition(
                input_points=torch.rand((30, 2)),
                output_points=torch.rand((30, 2)),
            ),
        }

    problem = SupervisedProblem()
    collector = Collector(problem)
    for v in collector.conditions_name.values():
        assert v in problem.conditions.keys()


def test_pinn_collector():
    def laplace_equation(input_, output_):
        force_term = torch.sin(input_.extract(["x"]) * torch.pi) * torch.sin(
            input_.extract(["y"]) * torch.pi
        )
        delta_u = laplacian(output_.extract(["u"]), input_)
        return delta_u - force_term

    my_laplace = Equation(laplace_equation)
    in_ = LabelTensor(
        torch.tensor([[0.0, 1.0]], requires_grad=True), ["x", "y"]
    )
    out_ = LabelTensor(torch.tensor([[0.0]], requires_grad=True), ["u"])

    class Poisson(SpatialProblem):
        output_variables = ["u"]
        spatial_domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})

        conditions = {
            "gamma1": Condition(
                domain=CartesianDomain({"x": [0, 1], "y": 1}),
                equation=FixedValue(0.0),
            ),
            "gamma2": Condition(
                domain=CartesianDomain({"x": [0, 1], "y": 0}),
                equation=FixedValue(0.0),
            ),
            "gamma3": Condition(
                domain=CartesianDomain({"x": 1, "y": [0, 1]}),
                equation=FixedValue(0.0),
            ),
            "gamma4": Condition(
                domain=CartesianDomain({"x": 0, "y": [0, 1]}),
                equation=FixedValue(0.0),
            ),
            "D": Condition(
                domain=CartesianDomain({"x": [0, 1], "y": [0, 1]}),
                equation=my_laplace,
            ),
            "data": Condition(input_points=in_, output_points=out_),
        }

        def poisson_sol(self, pts):
            return -(
                torch.sin(pts.extract(["x"]) * torch.pi)
                * torch.sin(pts.extract(["y"]) * torch.pi)
            ) / (2 * torch.pi**2)

        truth_solution = poisson_sol

    problem = Poisson()
    boundaries = ["gamma1", "gamma2", "gamma3", "gamma4"]
    problem.discretise_domain(10, "grid", domains=boundaries)
    problem.discretise_domain(10, "grid", domains="D")

    collector = Collector(problem)
    collector.store_fixed_data()
    collector.store_sample_domains()

    for k, v in problem.conditions.items():
        if isinstance(v, InputOutputPointsCondition):
            assert list(collector.data_collections[k].keys()) == [
                "input_points",
                "output_points",
            ]

    for k, v in problem.conditions.items():
        if isinstance(v, DomainEquationCondition):
            assert list(collector.data_collections[k].keys()) == [
                "input_points",
                "equation",
            ]


def test_supervised_graph_collector():
    pos = torch.rand((100, 3))
    x = [torch.rand((100, 3)) for _ in range(10)]
    graph = RadiusGraph(pos=pos, build_edge_attr=True, r=0.4)
    graph_list_1 = [graph(x=x_) for x_ in x]
    out_1 = torch.rand((10, 100, 3))

    pos = torch.rand((50, 3))
    x = [torch.rand((50, 3)) for _ in range(10)]
    graph = RadiusGraph(pos=pos, build_edge_attr=True, r=0.4)
    graph_list_2 = [graph(x=x_) for x_ in x]
    out_2 = torch.rand((10, 50, 3))

    class SupervisedProblem(AbstractProblem):
        output_variables = None
        conditions = {
            "data1": Condition(input_points=graph_list_1, output_points=out_1),
            "data2": Condition(input_points=graph_list_2, output_points=out_2),
        }

    problem = SupervisedProblem()
    collector = Collector(problem)
    collector.store_fixed_data()
    # assert all(collector._is_conditions_ready.values())
    for v in collector.conditions_name.values():
        assert v in problem.conditions.keys()


test_supervised_graph_collector()
