import torch
import pytest
from pina import LabelTensor
from pina.domain import EllipsoidDomain, Union

__dicts = [
    {"x": [0, 1], "y": [2.0, 3.5], "z": [0, 1.5]},
    {"x": [0, 1], "y": [2.0, 3.5], "z": 1.5},
    {"x": [0, 1], "y": 2.75, "z": 0.25},
    {"x": 0.0, "y": 2.5, "z": 1.0},
    {"x": (0, 1), "y": (0.0, 1.0)},
    {"x": (0, 1), "y": 0.0},
    {"x": 0.0, "y": 2.5},
    {"x": 0.0},
]


@pytest.mark.parametrize("dict", __dicts)
@pytest.mark.parametrize("sample_surface", [True, False])
def test_constructor(dict, sample_surface):
    EllipsoidDomain(ellipsoid_dict=dict, sample_surface=sample_surface)

    # Should fail if sample_surface is not a boolean
    with pytest.raises(ValueError):
        EllipsoidDomain(ellipsoid_dict=dict, sample_surface="invalid_value")

    # Should fail if the ellipsoid dictionary is not a dictionary
    with pytest.raises(TypeError):
        EllipsoidDomain(
            ellipsoid_dict=[("x", [0, 1]), ("y", [0, 1])],
            sample_surface=sample_surface,
        )

    # Should fail if the ellipsoid dictionary is empty
    with pytest.raises(ValueError):
        EllipsoidDomain(ellipsoid_dict={}, sample_surface=sample_surface)

    # Should fail if the value for a key is not numeric
    with pytest.raises(ValueError):
        EllipsoidDomain(
            ellipsoid_dict={"x": ["a", "b"]}, sample_surface=sample_surface
        )

    # Should fail if the value for a key is a list of lenght != 2
    with pytest.raises(ValueError):
        EllipsoidDomain(
            ellipsoid_dict={"x": [0, 1, 2]}, sample_surface=sample_surface
        )

    # Should fail if the range is invalid
    with pytest.raises(ValueError):
        EllipsoidDomain(
            ellipsoid_dict={"x": [1, 0]}, sample_surface=sample_surface
        )


@pytest.mark.parametrize("check_border", [True, False])
def test_is_inside(check_border):

    # Define points
    pt_in = LabelTensor(torch.tensor([[0.5, 0.5]]), ["x", "y"])
    pt_out = LabelTensor(torch.tensor([[1.5, 0.5]]), ["x", "y"])
    pt_border = LabelTensor(torch.tensor([[1.0, 0.5]]), ["x", "y"])

    # Define test domains
    domain = EllipsoidDomain(ellipsoid_dict=__dicts[4])

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


@pytest.mark.parametrize("dict", __dicts)
@pytest.mark.parametrize("sample_surface", [True, False])
def test_update(dict, sample_surface):

    # Define the domains
    domain_1 = EllipsoidDomain(
        ellipsoid_dict=dict, sample_surface=sample_surface
    )
    domain_2 = EllipsoidDomain(
        ellipsoid_dict={"new_var": [0, 1]}, sample_surface=sample_surface
    )
    domain_3 = EllipsoidDomain(
        ellipsoid_dict=dict | {"new_var": [0, 1]}, sample_surface=sample_surface
    )

    # Update domain_1 with domain_2
    updated_domain = domain_1.update(domain_2)

    # Check that domain_1 is now equal to domain_3
    assert updated_domain._fixed == domain_3._fixed
    assert updated_domain._range == domain_3._range

    # Should fail if trying to update with a different domain type (Union)
    with pytest.raises(TypeError):
        ellipsoid_domain = EllipsoidDomain({"x": [0, 1], "y": [0, 1]})
        other_domain = Union([ellipsoid_domain])
        updated_domain = ellipsoid_domain.update(other_domain)


@pytest.mark.parametrize("mode", ["random"])
@pytest.mark.parametrize("variables", ["all", "x", ["x"]])
@pytest.mark.parametrize("dicts", __dicts)
@pytest.mark.parametrize("sample_surface", [True, False])
def test_sample(mode, variables, dicts, sample_surface):

    # Sample from the domain and check that the points are inside
    num_samples = 5
    domain = EllipsoidDomain(dicts, sample_surface=sample_surface)
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


@pytest.mark.parametrize("dicts", __dicts)
@pytest.mark.parametrize("sample_surface", [True, False])
def test_partial(dicts, sample_surface):

    # Define the domain and get the boundary
    ellipsoid_domain = EllipsoidDomain(dicts, sample_surface=sample_surface)
    boundary = ellipsoid_domain.partial()

    # Checks
    assert isinstance(boundary, EllipsoidDomain)
    assert boundary._fixed == ellipsoid_domain._fixed
    assert boundary._range == ellipsoid_domain._range
    assert boundary._sample_surface == True
