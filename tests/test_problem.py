import torch
import pytest
from pina.problem.zoo import Poisson2DSquareProblem as Poisson
from pina import LabelTensor


def test_discretise_domain():
    n = 10
    poisson_problem = Poisson()
    boundaries = ['g1', 'g2', 'g3', 'g4']
    poisson_problem.discretise_domain(n, 'grid', domains=boundaries)
    for b in boundaries:
        assert poisson_problem.discretised_domains[b].shape[0] == n
    poisson_problem.discretise_domain(n, 'random', domains=boundaries)
    for b in boundaries:
        assert poisson_problem.discretised_domains[b].shape[0] == n

    poisson_problem.discretise_domain(n, 'grid', domains=['D'])
    assert poisson_problem.discretised_domains['D'].shape[0] == n ** 2
    poisson_problem.discretise_domain(n, 'random', domains=['D'])
    assert poisson_problem.discretised_domains['D'].shape[0] == n

    poisson_problem.discretise_domain(n, 'latin', domains=['D'])
    assert poisson_problem.discretised_domains['D'].shape[0] == n

    poisson_problem.discretise_domain(n, 'lh', domains=['D'])
    assert poisson_problem.discretised_domains['D'].shape[0] == n

    poisson_problem.discretise_domain(n)


'''
def test_sampling_few_variables():
    n = 10
    poisson_problem = Poisson()
    poisson_problem.discretise_domain(n,
                                      'grid',
                                      domains=['D'],
                                      variables=['x'])
    assert poisson_problem.discretised_domains['D'].shape[1] == 1
'''


def test_variables_correct_order_sampling():
    n = 10
    poisson_problem = Poisson()
    poisson_problem.discretise_domain(n,
                                      'grid',
                                      domains=['D'])
    assert poisson_problem.discretised_domains['D'].labels == sorted(
        poisson_problem.input_variables)

    poisson_problem.discretise_domain(n, 'grid', domains=['D'])
    assert poisson_problem.discretised_domains['D'].labels == sorted(
        poisson_problem.input_variables)


def test_add_points():
    poisson_problem = Poisson()
    poisson_problem.discretise_domain(0,
                                      'random',
                                      domains=['D'])
    new_pts = LabelTensor(torch.tensor([[0.5, -0.5]]), labels=['x', 'y'])
    poisson_problem.add_points({'D': new_pts})
    assert torch.isclose(poisson_problem.discretised_domains['D'].extract('x'),
                         new_pts.extract('x'))
    assert torch.isclose(poisson_problem.discretised_domains['D'].extract('y'),
                         new_pts.extract('y'))
