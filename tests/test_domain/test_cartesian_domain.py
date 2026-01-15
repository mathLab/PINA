import torch
import pytest
from pina import LabelTensor
from pina.domain import CartesianDomain, Union

__dicts = [
    {"x": [0, 1], "y": [2.0, 3.5], "z": [0, 1.5]},
    {"x": [0, 1], "y": [2.0, 3.5], "z": 1.5},
    {"x": [0, 1], "y": 2.75, "z": 0.25},
    {"x": 0.0, "y": 2.5, "z": 1.0},
    {"x": (0, 1), "y": (0.0, 1.0)},
    {"x": (0, 1), "y": 0.5},
    {"x": 0.0, "y": 2.5},
    {"x": 0.0},
]


@pytest.mark.parametrize("dict", __dicts)
def test_constructor(dict):
    CartesianDomain(dict)

    # Should fail if the cartesian dictionary is not a dictionary
    with pytest.raises(TypeError):
        CartesianDomain([("x", [0, 1]), ("y", [0, 1])])

    # Should fail if the cartesian dictionary is empty
    with pytest.raises(ValueError):
        CartesianDomain({})

    # Should fail if the value for a key is not numeric
    with pytest.raises(ValueError):
        CartesianDomain({"x": ["a", "b"]})

    # Should fail if the value for a key is a list of lenght != 2
    with pytest.raises(ValueError):
        CartesianDomain({"x": [0, 1, 2]})

    # Should fail if the range is invalid
    with pytest.raises(ValueError):
        CartesianDomain({"x": [1, 0]})


@pytest.mark.parametrize("check_border", [True, False])
def test_is_inside(check_border):

    # Define points
    pt_in = LabelTensor(torch.tensor([[0.5, 0.5]]), ["x", "y"])
    pt_out = LabelTensor(torch.tensor([[1.5, 0.5]]), ["x", "y"])
    pt_border = LabelTensor(torch.tensor([[1.0, 0.5]]), ["x", "y"])

    # Define test domains
    domain_1 = CartesianDomain(__dicts[4])
    domain_2 = CartesianDomain(__dicts[5])

    # Expected results
    truth_1 = [True, False, True] if check_border else [True, False, False]
    truth_2 = [True, False, True] if check_border else [True, False, False]

    # Checks
    for pt, exp_1, exp_2 in zip([pt_in, pt_out, pt_border], truth_1, truth_2):
        assert domain_1.is_inside(pt, check_border=check_border) == exp_1
        assert domain_2.is_inside(pt, check_border=check_border) == exp_2

    # Should fail if point is not a LabelTensor
    with pytest.raises(ValueError):
        domain_1.is_inside(torch.Tensor([0.5, 0.5]), check_border=check_border)

    # Should fail if the labels of the point differ from the domain
    with pytest.raises(ValueError):
        pt = LabelTensor(torch.Tensor([0.5, 0.5]), ["a", "b"])
        domain_1.is_inside(pt, check_border=check_border)


@pytest.mark.parametrize("dict", __dicts)
def test_update(dict):

    # Define the domains
    domain_1 = CartesianDomain(dict)
    domain_2 = CartesianDomain({"new_var": [0, 1]})
    domain_3 = CartesianDomain(dict | {"new_var": [0, 1]})

    # Update domain_1 with domain_2
    updated_domain = domain_1.update(domain_2)

    # Check that domain_1 is now equal to domain_3
    assert updated_domain._fixed == domain_3._fixed
    assert updated_domain._range == domain_3._range

    # Should fail if trying to update with a different domain type (Union)
    with pytest.raises(TypeError):
        cartesian_domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})
        other_domain = Union([cartesian_domain])
        updated_domain = cartesian_domain.update(other_domain)


@pytest.mark.parametrize("mode", ["grid", "random", "lh", "chebyshev"])
@pytest.mark.parametrize("variables", ["all", "x", ["x"]])
@pytest.mark.parametrize("dicts", __dicts)
def test_sample(mode, variables, dicts):

    # Sample from the domain
    num_samples = 5
    domain = CartesianDomain(dicts)
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


@pytest.mark.parametrize("dicts", __dicts)
def test_partial(dicts):

    # Define the domain and get the boundary
    cartesian_domain = CartesianDomain(dicts)
    boundary = cartesian_domain.partial()
    faces = boundary.geometries

    # Checks
    assert isinstance(boundary, Union)
    assert len(faces) == 2 * len(cartesian_domain._range)
    assert all(isinstance(f, CartesianDomain) for f in faces)

    # Iterate over the faces
    for face in faces:

        # Each face should differ from the original domain by exactly 1 variable
        diff_keys = [
            k
            for k in face.variables
            if cartesian_domain.domain_dict[k] != face.domain_dict[k]
        ]

        # Check that only one variable differs
        assert len(diff_keys) == 1

        # Check that the differing variable is fixed to one of the bounds
        assert (
            face.domain_dict[diff_keys[0]]
            in cartesian_domain._range[diff_keys[0]]
        )
