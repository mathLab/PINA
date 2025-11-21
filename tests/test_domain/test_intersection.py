import torch
import pytest
from pina import LabelTensor
from pina.domain import (
    Intersection,
    EllipsoidDomain,
    CartesianDomain,
    SimplexDomain,
)

# Define the domains for testing
cartesian_1 = CartesianDomain({"x": [0, 2], "y": [0, 2]})
cartesian_2 = CartesianDomain({"x": [0, 2], "y": [0, 2], "z": [0, 2]})

ellipsoid_1 = EllipsoidDomain({"x": [-1, 1], "y": [-1, 1]})
ellipsoid_2 = EllipsoidDomain({"x": [0, 1], "y": [-1, 1], "z": [1, 3]})

simplex_1 = SimplexDomain(
    [
        LabelTensor(torch.tensor([[0, 0]]), labels=["x", "y"]),
        LabelTensor(torch.tensor([[1, 0]]), labels=["x", "y"]),
        LabelTensor(torch.tensor([[0, 1]]), labels=["x", "y"]),
    ]
)
simplex_2 = SimplexDomain(
    [
        LabelTensor(torch.tensor([[0, 0, 0]]), labels=["x", "y", "z"]),
        LabelTensor(torch.tensor([[2, 0, 0]]), labels=["x", "y", "z"]),
        LabelTensor(torch.tensor([[0, 2, 0]]), labels=["x", "y", "z"]),
        LabelTensor(torch.tensor([[0, 0, 2]]), labels=["x", "y", "z"]),
    ]
)

# Define the geometries
__geometries = [
    [cartesian_1, ellipsoid_1],
    [cartesian_2, ellipsoid_2],
    [cartesian_1, simplex_1],
    [cartesian_2, simplex_2],
    [ellipsoid_1, simplex_1],
    [ellipsoid_2, simplex_2],
    [cartesian_1, ellipsoid_1, simplex_1],
    [cartesian_2, ellipsoid_2, simplex_2],
]


@pytest.mark.parametrize("geometries", __geometries)
def test_constructor(geometries):
    Intersection(geometries)

    # Should fail if geometries is not a list or a tuple
    with pytest.raises(TypeError):
        Intersection({cartesian_1, ellipsoid_1})

    # Should fail if the elements of geometries are not BaseDomain instances
    with pytest.raises(ValueError):
        Intersection([{"x": [0, 1], "y": [0, 1]}, {"x": [1, 2], "y": [0, 1]}])

    # Should fail if the dimensions of the geometries are not consistent
    with pytest.raises(NotImplementedError):
        Intersection([cartesian_1, cartesian_2])


@pytest.mark.parametrize("check_border", [True, False])
def test_is_inside(check_border):

    # Define points
    pt_in = LabelTensor(torch.tensor([[0.2, 0.2]]), ["x", "y"])
    pt_out = LabelTensor(torch.tensor([[-0.2, -0.2]]), ["x", "y"])
    pt_border = LabelTensor(torch.tensor([[0, 0]]), ["x", "y"])

    # Intersection
    intersection = Intersection(__geometries[0])

    # Expected results
    truth = [True, False, True] if check_border else [True, False, False]

    # Checks
    for pt, exp in zip([pt_in, pt_out, pt_border], truth):
        assert intersection.is_inside(pt, check_border=check_border) == exp

    # Should fail if point is not a LabelTensor
    with pytest.raises(ValueError):
        intersection.is_inside(
            torch.Tensor([0.5, 0.5]), check_border=check_border
        )

    # Should fail if the labels of the point differ from the domain
    with pytest.raises(ValueError):
        pt = LabelTensor(torch.Tensor([0.5, 0.5]), ["a", "b"])
        intersection.is_inside(pt, check_border=check_border)


@pytest.mark.parametrize("domain_class", [CartesianDomain, EllipsoidDomain])
def test_update(domain_class):

    # Define the intersection
    domain_1 = domain_class({"x": [0, 1], "y": [0, 1]})
    domain_2 = domain_class({"x": [0.5, 1.5], "y": [0, 2]})
    intersection = Intersection([domain_1, domain_2])

    # Update the intersection with another valid domain
    domain_3 = domain_class({"t": [0, 1], "w": 0})
    updated_intersection = intersection.update(domain_3)

    # Check that the intersection has been updated correctly
    assert len(updated_intersection.geometries) == 2
    assert updated_intersection.variables == sorted(["x", "y", "t", "w"])
    for i, g in enumerate(updated_intersection.geometries):
        assert g._range == {
            **intersection.geometries[i]._range,
            **domain_3._range,
        }
        assert g._fixed == {
            **intersection.geometries[i]._fixed,
            **domain_3._fixed,
        }

    # Should fail if trying to update the intersection of different geometry types
    with pytest.raises(NotImplementedError):
        intersection = Intersection(__geometries[0])
        intersection.update(simplex_1)

    # Should fail if trying to update with a different domain type
    with pytest.raises(TypeError):
        intersection = Intersection(
            CartesianDomain({"x": [0, 1], "y": [0, 1]}),
            CartesianDomain({"x": [1, 2], "y": [0, 1]}),
        )
        other_domain = EllipsoidDomain({"x": [-1, 1], "y": [-1, 1]})
        intersection.update(other_domain)


def test_partial():
    with pytest.raises(NotImplementedError):
        intersection = Intersection(__geometries[0])
        intersection.partial()


@pytest.mark.parametrize("variables", ["all", "x", ["x"]])
@pytest.mark.parametrize("geometries", __geometries)
def test_sample(variables, geometries):

    # Define the domain
    num_samples = 5
    domain = Intersection(geometries)

    # Iterate over modes (dependent on the domain types)
    for mode in domain.sample_modes:

        # Sample from the domain
        pts = domain.sample(num_samples, mode=mode, variables=variables)

        # Labels and number of samples
        labels = sorted(variables if variables != "all" else domain.variables)
        if mode in ["grid", "chebyshev"]:
            num_range_vars = len([k for k in labels if k in domain._range])
            num_samples = num_samples ** (num_range_vars or 1)

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
