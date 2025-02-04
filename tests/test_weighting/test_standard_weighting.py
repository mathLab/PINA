import pytest
import torch

from pina.solvers import PINN
from pina.problem.zoo import Poisson2DSquareProblem
from pina.loss import ScalarWeighting

problem = Poisson2DSquareProblem()
condition_names = problem.conditions.keys()
print(problem.conditions.keys())

@pytest.mark.parametrize("weights",
                         [1, 1., dict(zip(condition_names, [1]*len(condition_names)))])
def test_constructor(weights):
    ScalarWeighting(weights=weights)

@pytest.mark.parametrize("weights", ['a', [1,2,3]])
def test_wrong_constructor(weights):
    with pytest.raises(ValueError):
        ScalarWeighting(weights=weights)

@pytest.mark.parametrize("weights",
                         [1, 1., dict(zip(condition_names, [1]*len(condition_names)))])
def test_aggregate(weights):
    weighting = ScalarWeighting(weights=weights)
    losses = dict(zip(condition_names, [torch.randn(1) for _ in range(len(condition_names))]))
    # this raises an error because no link between solver and weighting is created
    with pytest.raises(TypeError):
        weighting.aggregate(losses=losses)
    # when the solver is initialized the link between weighting and problem is
    # created
    solver = PINN(
                problem=problem,
                model=torch.nn.Linear(
                    len(problem.input_variables),
                    len(problem.output_variables)),
                weighting=weighting)
    weighting.aggregate(losses=losses)