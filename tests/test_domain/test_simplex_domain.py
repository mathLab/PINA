import torch
import pytest
from pina import LabelTensor
from pina.domain import SimplexDomain, Union

__matrices = [
    [
        LabelTensor(torch.tensor([[0, 0]]), labels=["x", "y"]),
        LabelTensor(torch.tensor([[0, 1]]), labels=["x", "y"]),
        LabelTensor(torch.tensor([[1, 0]]), labels=["x", "y"]),
    ],
    [
        LabelTensor(torch.tensor([[0, 0, 0]]), labels=["x", "y", "z"]),
        LabelTensor(torch.tensor([[0, 1, 0]]), labels=["x", "y", "z"]),
        LabelTensor(torch.tensor([[1, 0, 0]]), labels=["x", "y", "z"]),
        LabelTensor(torch.tensor([[0, 0, 1]]), labels=["x", "y", "z"]),
    ],
    [
        LabelTensor(torch.tensor([[0, 0, 0, 0]]), labels=["w", "x", "y", "z"]),
        LabelTensor(torch.tensor([[1, 0, 0, 0]]), labels=["w", "x", "y", "z"]),
        LabelTensor(torch.tensor([[0, 1, 0, 0]]), labels=["w", "x", "y", "z"]),
        LabelTensor(torch.tensor([[0, 0, 1, 0]]), labels=["w", "x", "y", "z"]),
        LabelTensor(torch.tensor([[0, 0, 0, 1]]), labels=["w", "x", "y", "z"]),
    ],
]


@pytest.mark.parametrize("matrices", __matrices)
@pytest.mark.parametrize("sample_surface", [True, False])
def test_constructor(matrices, sample_surface):
    SimplexDomain(simplex_matrix=matrices, sample_surface=sample_surface)

    # Should fail if simplex_matrix is not a list or tuple
    with pytest.raises(ValueError):
        SimplexDomain(simplex_matrix="invalid", sample_surface=sample_surface)

    # Should fail if any element of simplex_matrix is not a LabelTensor
    with pytest.raises(ValueError):
        invalid_mat = [
            LabelTensor(torch.tensor([[0, 0]]), labels=["x", "y"]),
            LabelTensor(torch.tensor([[0, 1]]), labels=["x", "y"]),
            torch.tensor([[1, 0]]),
        ]
        SimplexDomain(simplex_matrix=invalid_mat, sample_surface=sample_surface)

    # Should fail if sample_surface is not a boolean
    with pytest.raises(ValueError):
        SimplexDomain(simplex_matrix=matrices, sample_surface="invalid_value")

    # Should fail if the labels of the vertices do not match
    with pytest.raises(ValueError):
        invalid_mat = [
            LabelTensor(torch.tensor([[0, 0]]), labels=["x", "y"]),
            LabelTensor(torch.tensor([[0, 1]]), labels=["x", "y"]),
            LabelTensor(torch.tensor([[1, 0]]), labels=["a", "b"]),
        ]
        SimplexDomain(simplex_matrix=invalid_mat, sample_surface=sample_surface)

    # Should fail if the number of vertices is not equal to dimension + 1
    with pytest.raises(ValueError):
        invalid_mat = [
            LabelTensor(torch.tensor([[0, 0]]), labels=["x", "y"]),
            LabelTensor(torch.tensor([[0, 1]]), labels=["x", "y"]),
            LabelTensor(torch.tensor([[1, 0]]), labels=["x", "y"]),
            LabelTensor(torch.tensor([[1, 1]]), labels=["x", "y"]),
        ]
        SimplexDomain(simplex_matrix=invalid_mat, sample_surface=sample_surface)


@pytest.mark.parametrize("check_border", [True, False])
def test_is_inside(check_border):

    # Define points
    pt_in = LabelTensor(torch.tensor([[0.2, 0.2]]), ["x", "y"])
    pt_out = LabelTensor(torch.tensor([[1.5, 0.2]]), ["x", "y"])
    pt_border = LabelTensor(torch.tensor([[0.8, 0.2]]), ["x", "y"])

    # Define test domains
    domain = SimplexDomain(simplex_matrix=__matrices[0])

    # Expected results
    truth = [True, False, True] if check_border else [True, False, False]

    # Checks
    for pt, exp in zip([pt_in, pt_out, pt_border], truth):
        assert domain.is_inside(pt, check_border=check_border) == exp

    # Should fail if point is not a LabelTensor
    with pytest.raises(ValueError):
        domain.is_inside(torch.Tensor([0.5, 0.5]), check_border=check_border)

    # Should fail if the labels of the point differ from the domain
    with pytest.raises(ValueError):
        pt = LabelTensor(torch.Tensor([0.5, 0.5]), ["a", "b"])
        domain.is_inside(pt, check_border=check_border)


@pytest.mark.parametrize("matrices", __matrices)
@pytest.mark.parametrize("sample_surface", [True, False])
def test_update(matrices, sample_surface):

    # Define the domains
    domain_1 = SimplexDomain(
        simplex_matrix=matrices, sample_surface=sample_surface
    )
    domain_2 = SimplexDomain(
        simplex_matrix=[
            LabelTensor(torch.tensor([[0, 0]]), labels=["x", "y"]),
            LabelTensor(torch.tensor([[1, 0]]), labels=["x", "y"]),
            LabelTensor(torch.tensor([[0, 1]]), labels=["x", "y"]),
        ],
        sample_surface=sample_surface,
    )

    # Update domain_1 with domain_2
    updated_domain = domain_1.update(domain_2)

    # Check that domain_1 is now equal to domain_2
    assert updated_domain.variables == domain_2.variables
    for v1, v2 in zip(updated_domain._vert_matrix, domain_2._vert_matrix):
        assert torch.allclose(v1.tensor, v2.tensor, atol=1e-12, rtol=0)

    # Should fail if trying to update with a different domain type (Union)
    with pytest.raises(TypeError):
        other_domain = Union([domain_2])
        updated_domain = domain_1.update(other_domain)


@pytest.mark.parametrize("mode", ["random"])
@pytest.mark.parametrize("variables", ["all", "x", ["x"]])
@pytest.mark.parametrize("matrices", __matrices)
@pytest.mark.parametrize("sample_surface", [True, False])
def test_sample(mode, variables, matrices, sample_surface):

    # Sample from the domain and check that the points are inside
    num_samples = 5
    domain = SimplexDomain(matrices, sample_surface=sample_surface)
    pts = domain.sample(num_samples, mode=mode, variables=variables)

    # Labels
    labels = sorted(variables if variables != "all" else domain.variables)

    # Checks
    assert pts.shape == (num_samples, len(labels))
    assert pts.labels == labels

    # Should fail if n is not a positive integer
    with pytest.raises(AssertionError):
        domain.sample(0, mode=mode, variables=variables)

    # Should fail if the mode is not recognized
    with pytest.raises(ValueError):
        domain.sample(1, mode="invalid_mode", variables=variables)

    # Should fail if the variables are invalid
    with pytest.raises(ValueError):
        domain.sample(1, mode=mode, variables=123)

    # Should fail if the variables are unknown
    with pytest.raises(ValueError):
        domain.sample(1, mode=mode, variables=["invalid_var"])


@pytest.mark.parametrize("matrices", __matrices)
@pytest.mark.parametrize("sample_surface", [True, False])
def test_partial(matrices, sample_surface):

    # Define the domain and get the boundary
    simplex_domain = SimplexDomain(matrices, sample_surface=sample_surface)
    boundary = simplex_domain.partial()

    # Checks
    assert isinstance(boundary, SimplexDomain)
    assert boundary._sample_surface == True
    for v1, v2 in zip(simplex_domain._vert_matrix, boundary._vert_matrix):
        assert torch.allclose(v1.tensor, v2.tensor, atol=1e-12, rtol=0)
