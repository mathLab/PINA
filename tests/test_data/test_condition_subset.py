import torch
import pytest
from pina.equation.zoo import FixedValue
from pina import Condition, LabelTensor
from pina.domain import CartesianDomain
from pina.data import _ConditionSubset

# Define an equation and a domain for testing purposes
equation = FixedValue(value=0.0)
domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})

# Define input and target tensors for testing purposes
n_val, n_dim = 5, 2
input_tensor = torch.rand(n_val, n_dim)
input_label_tensor = LabelTensor(torch.rand(n_val, n_dim), labels=["x", "y"])
target_tensor = torch.rand(n_val, n_dim)
cond_vars = torch.rand(n_val, 1)

# Define conditions for testing purposes
# Domain - equation condition is not tested as __get_item__ is not implemented
input_target_cond = Condition(input=input_tensor, target=target_tensor)
input_equation_cond = Condition(input=input_label_tensor, equation=equation)
data_cond = Condition(input=input_tensor, conditional_variables=cond_vars)

# Define indexes for testing purposes
indices = torch.randperm(n_val).tolist()


@pytest.mark.parametrize("automatic_batching", [True, False])
@pytest.mark.parametrize("indices", [indices[:3], indices[:2]])
@pytest.mark.parametrize(
    "condition", [input_target_cond, input_equation_cond, data_cond]
)
def test_constructor(condition, automatic_batching, indices):

    # Initialize the condition subset
    subset = _ConditionSubset(
        condition=condition,
        indices=indices,
        automatic_batching=automatic_batching,
    )

    # Verify that the attributes are correctly assigned
    assert subset.condition is condition
    assert subset.indices == indices
    assert subset.automatic_batching is automatic_batching
    assert subset.dataset_length == len(indices)
    assert subset.iterable_length == len(indices)


@pytest.mark.parametrize("automatic_batching", [True, False])
@pytest.mark.parametrize("indices", [indices[:3], indices[:2]])
@pytest.mark.parametrize(
    "condition", [input_target_cond, input_equation_cond, data_cond]
)
def test_len(condition, automatic_batching, indices):

    # Initialize the condition subset
    subset = _ConditionSubset(
        condition=condition,
        indices=indices,
        automatic_batching=automatic_batching,
    )

    # Verify that the length of the subset is correctly computed
    assert len(subset) == len(indices)


@pytest.mark.parametrize("automatic_batching", [True, False])
@pytest.mark.parametrize("indices", [indices[:3], indices[:2]])
@pytest.mark.parametrize(
    "condition", [input_target_cond, input_equation_cond, data_cond]
)
def test_get_item(condition, automatic_batching, indices):

    # Initialize the condition subset
    subset = _ConditionSubset(
        condition=condition,
        indices=indices,
        automatic_batching=automatic_batching,
    )

    # Verify that the correct data is returned for each index in the subset
    for local_idx in range(len(indices)):

        # Retrieve the true dataset index
        true_idx = indices[local_idx]

        # If automatic batching, check data equivalence
        if automatic_batching:

            # Save actual and expected data for debugging purposes
            actual_data = subset[local_idx].data
            expected_data = condition[true_idx].data

            # Check that the keys of the returned data match
            assert actual_data.keys() == expected_data.keys()

            # Check that the values of the returned data are equal
            for key in actual_data:
                assert torch.equal(actual_data[key], expected_data[key])

        # Otherwise, check that the raw dataset index is returned
        else:
            assert subset[local_idx] == true_idx

    # Check cyclic indexing
    cyclic_idx = len(indices)
    true_idx = indices[0]

    # If automatic batching, check data equivalence for cyclic index
    if automatic_batching:

        # Check that the keys of the returned data match
        assert subset[cyclic_idx].data.keys() == condition[true_idx].data.keys()

        # Check that the values of the returned data are equal
        for key in actual_data:
            assert torch.equal(actual_data[key], expected_data[key])

    # Otherwise, check that the raw dataset index is returned for cyclic index
    else:
        assert subset[cyclic_idx] == true_idx


@pytest.mark.parametrize("automatic_batching", [True, False])
@pytest.mark.parametrize(
    "condition",
    [input_target_cond, input_equation_cond, data_cond],
)
def test_get_all_data(condition, automatic_batching):

    # Initialize the condition subset
    subset = _ConditionSubset(
        condition=condition,
        indices=indices,
        automatic_batching=automatic_batching,
    )

    # Retrieve all data from the subset and check that it matches expected data
    data = subset.get_all_data()
    expected = condition[indices]

    # Check that the keys of the returned data match
    assert data.keys == expected.keys

    # Check that the values of the returned data are equal
    for key in data.keys:
        assert torch.equal(data.data[key], expected.data[key])
