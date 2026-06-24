import torch
import pytest
from pina import LabelTensor
from pina.problem.zoo import Poisson2DSquareProblem as Poisson


# Define sampling rules
rule1 = {
    "x": {"n": 10, "mode": "random"},
    "y": {"n": 5, "mode": "grid"},
}
rule2 = {
    "x": {"n": 5, "mode": "lh"},
    "y": {"n": 10, "mode": "chebyshev"},
}


@pytest.mark.parametrize("n", [2, 5])
@pytest.mark.parametrize("mode", ["grid", "random", "latin", "chebyshev", "lh"])
@pytest.mark.parametrize("domains", ["boundary", "D", ["boundary", "D"], None])
@pytest.mark.parametrize("sample_rules", [None, rule1, rule2])
def test_discretise_domain(n, mode, domains, sample_rules):

    # Define the problem
    poisson_problem = Poisson()

    # Discretise domains
    poisson_problem.discretise_domain(
        n=n, mode=mode, domains=domains, sample_rules=sample_rules
    )

    # Transform domains to list for consistent processing
    _as_list = lambda x: [x] if isinstance(x, str) else x
    d_list = domains if domains is not None else ["boundary", "D"]
    d_list = _as_list(d_list)

    # Check that the discretised domains have the expected number of points
    for d in d_list:

        # Compute expected number of points if sample rules are provided
        if sample_rules is not None:
            n_tot = sample_rules["x"]["n"] * sample_rules["y"]["n"]

        # Otherwise, expect n pts or n^2 based on domain and mode
        else:
            n_tot = n**2 if mode in ["grid", "chebyshev"] and d == "D" else n

        # Check that the number of samples matches the expected number
        assert poisson_problem.discretised_domains[d].shape[0] == n_tot

        # Check labels of the discretised domains
        assert poisson_problem.discretised_domains[d].labels == sorted(
            poisson_problem.input_variables
        )

    # Should fail if n is not a positive integer when sample rules not provided
    if sample_rules is None:
        with pytest.raises(AssertionError):
            poisson_problem.discretise_domain(
                n=-1, mode=mode, domains=domains, sample_rules=sample_rules
            )

    # Should fail if mode is not a string
    with pytest.raises(ValueError):
        poisson_problem.discretise_domain(
            n=n, mode=123, domains=domains, sample_rules=sample_rules
        )

    # Should fail if domains is not a string or a list of strings
    with pytest.raises(ValueError):
        poisson_problem.discretise_domain(
            n=n, mode=mode, domains=123, sample_rules=sample_rules
        )

    # Should fail if sample rules is not a dictionary
    with pytest.raises(ValueError):
        poisson_problem.discretise_domain(
            n=n, mode=mode, domains=domains, sample_rules="not_a_dict"
        )

    # Should fail if the keys of sample rules do not match the input variables
    with pytest.raises(ValueError):
        wrong_sample_rules = {"wrong_var": {"n": 10, "mode": "random"}}
        poisson_problem.discretise_domain(
            n=n, mode=mode, domains=domains, sample_rules=wrong_sample_rules
        )

    # Should fail if the rules do not contain both 'n' and 'mode' keys
    with pytest.raises(ValueError):
        incomplete_sample_rules = {"x": {"n": 10}, "y": {"mode": "random"}}
        poisson_problem.discretise_domain(
            n=n,
            mode=mode,
            domains=domains,
            sample_rules=incomplete_sample_rules,
        )


@pytest.mark.parametrize("domains", ["boundary", "D", ["boundary", "D"], None])
def test_add_points(domains):

    # Store initial number of points in the domains and point to add
    n_init, n_add = 5, 3
    n_tot = n_init + n_add

    # Define the problem and discretise the domain
    poisson_problem = Poisson()
    poisson_problem.discretise_domain(n=n_init, mode="random", domains=domains)
    vars = poisson_problem.input_variables

    # Transform domains to list for consistent processing
    _as_list = lambda x: [x] if isinstance(x, str) else x
    d_list = domains if domains is not None else ["boundary", "D"]
    d_list = _as_list(d_list)

    # Iterate over the domains and add points to each
    for d in d_list:

        # Add new points to the domain
        new_pts = LabelTensor(torch.rand(n_add, len(vars)), labels=vars)
        poisson_problem.add_points({d: new_pts})

        # Assert that the number of points in the domain is correct
        assert poisson_problem.discretised_domains[d].shape[0] == n_tot

        # Assert that the new points are in the domain
        assert torch.allclose(
            poisson_problem.discretised_domains[d]["x"][-n_add:], new_pts["x"]
        )
        assert torch.allclose(
            poisson_problem.discretised_domains[d]["y"][-n_add:], new_pts["y"]
        )

    # Should fail if new points is not a dictionary
    with pytest.raises(ValueError):
        poisson_problem.add_points("not_a_dict")

    # Should fail if any of the values in new points is not a LabelTensor
    with pytest.raises(ValueError):
        poisson_problem.add_points({d_list[0]: torch.rand(n_add, len(vars))})

    # Should fail if any of the keys does not match any of the existing domains
    with pytest.raises(ValueError):
        poisson_problem.add_points(
            {
                "not_a_domain": LabelTensor(
                    torch.rand(n_add, len(vars)), labels=vars
                )
            }
        )

    # Should fail if any of the domains has not been discretised yet
    with pytest.raises(ValueError):
        poisson_problem = Poisson()
        poisson_problem.discretise_domain(n=n_init, mode="random", domains="D")
        poisson_problem.add_points(
            {"boundary": LabelTensor(torch.rand(n_add, len(vars)), labels=vars)}
        )
