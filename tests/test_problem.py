import torch
import pytest
from pina.problem.zoo import Poisson2DSquareProblem as Poisson
from pina import LabelTensor
from pina.domain import Union, CartesianDomain, EllipsoidDomain
from pina.condition import (
    Condition,
    InputTargetCondition,
    DomainEquationCondition,
)


def test_discretise_domain():
    n = 10
    poisson_problem = Poisson()

    poisson_problem.discretise_domain(n, "grid", domains="boundary")
    assert poisson_problem.discretised_domains["boundary"].shape[0] == n

    poisson_problem.discretise_domain(n, "random", domains="boundary")
    assert poisson_problem.discretised_domains["boundary"].shape[0] == n

    poisson_problem.discretise_domain(n, "grid", domains=["D"])
    assert poisson_problem.discretised_domains["D"].shape[0] == n**2

    poisson_problem.discretise_domain(n, "random", domains=["D"])
    assert poisson_problem.discretised_domains["D"].shape[0] == n

    poisson_problem.discretise_domain(n, "latin", domains=["D"])
    assert poisson_problem.discretised_domains["D"].shape[0] == n

    poisson_problem.discretise_domain(n, "lh", domains=["D"])
    assert poisson_problem.discretised_domains["D"].shape[0] == n

    poisson_problem.discretise_domain(n)


def test_variables_correct_order_sampling():
    n = 10
    poisson_problem = Poisson()
    poisson_problem.discretise_domain(n, "grid", domains=["D"])
    assert poisson_problem.discretised_domains["D"].labels == sorted(
        poisson_problem.input_variables
    )

    poisson_problem.discretise_domain(n, "grid", domains=["D"])
    assert poisson_problem.discretised_domains["D"].labels == sorted(
        poisson_problem.input_variables
    )


def test_input_pts():
    n = 10
    poisson_problem = Poisson()
    poisson_problem.discretise_domain(n, "grid")
    assert sorted(list(poisson_problem.input_pts.keys())) == sorted(
        list(poisson_problem.conditions.keys())
    )


def test_collected_data():
    n = 10
    poisson_problem = Poisson()
    poisson_problem.discretise_domain(n, "grid")
    assert sorted(list(poisson_problem.collected_data.keys())) == sorted(
        list(poisson_problem.conditions.keys())
    )


def test_add_points():
    poisson_problem = Poisson()
    poisson_problem.discretise_domain(1, "random", domains=["D"])
    new_pts = LabelTensor(torch.tensor([[0.5, -0.5]]), labels=["x", "y"])
    poisson_problem.add_points({"D": new_pts})
    assert torch.allclose(
        poisson_problem.discretised_domains["D"]["x"][-1],
        new_pts["x"],
    )
    assert torch.allclose(
        poisson_problem.discretised_domains["D"]["y"][-1],
        new_pts["y"],
    )


@pytest.mark.parametrize("mode", ["random", "grid"])
def test_custom_sampling_logic(mode):
    poisson_problem = Poisson()
    sampling_rules = {
        "x": {"n": 100, "mode": mode},
        "y": {"n": 50, "mode": mode},
    }
    poisson_problem.discretise_domain(sample_rules=sampling_rules, domains="D")
    assert poisson_problem.discretised_domains["D"].shape[0] == 100 * 50
    assert poisson_problem.discretised_domains["D"].labels == ["x", "y"]


@pytest.mark.parametrize("mode", ["random", "grid"])
def test_wrong_custom_sampling_logic(mode):
    d2 = CartesianDomain({"x": [1, 2], "y": [0, 1]})
    poisson_problem = Poisson()
    poisson_problem.domains["D"] = Union([poisson_problem.domains["D"], d2])
    sampling_rules = {
        "x": {"n": 100, "mode": mode},
        "y": {"n": 50, "mode": mode},
    }
    with pytest.raises(RuntimeError):
        poisson_problem.domains["new"] = EllipsoidDomain({"x": [0, 1]})
        poisson_problem.discretise_domain(sample_rules=sampling_rules)

    # Necessary cleanup
    if "new" in poisson_problem.domains:
        del poisson_problem.domains["new"]


def test_aggregate_data():
    poisson_problem = Poisson()
    poisson_problem.conditions["data"] = Condition(
        input=LabelTensor(torch.tensor([[0.0, 1.0]]), labels=["x", "y"]),
        target=LabelTensor(torch.tensor([[0.0]]), labels=["u"]),
    )
    poisson_problem.discretise_domain(1, "random", domains="all")
    poisson_problem.collect_data()
    assert isinstance(poisson_problem.collected_data, dict)
    for name, conditions in poisson_problem.conditions.items():
        assert name in poisson_problem.collected_data.keys()
        if isinstance(conditions, InputTargetCondition):
            assert "input" in poisson_problem.collected_data[name].keys()
            assert "target" in poisson_problem.collected_data[name].keys()
        elif isinstance(conditions, DomainEquationCondition):
            assert "input" in poisson_problem.collected_data[name].keys()
            assert "target" not in poisson_problem.collected_data[name].keys()
            assert "equation" in poisson_problem.collected_data[name].keys()
