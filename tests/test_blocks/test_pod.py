import torch
import pytest

from pina.model.block.pod_block import PODBlock

x = torch.linspace(-1, 1, 100)
toy_snapshots = torch.vstack(
    [torch.exp(-(x**2)) * c for c in torch.linspace(0, 1, 10)]
)


def test_constructor():
    pod = PODBlock(2)
    pod = PODBlock(2, True)
    pod = PODBlock(2, False)
    with pytest.raises(TypeError):
        pod = PODBlock()


@pytest.mark.parametrize("rank", [1, 2, 10])
def test_fit(rank, scale):
    pod = PODBlock(rank, scale)
    assert pod._basis == None
    assert pod.basis == None
    assert pod._scaler == None
    assert pod._singular_values == None
    assert pod.singular_values == None
    assert pod.rank == rank
    assert pod.scale_coefficients == scale


@pytest.mark.parametrize("scale", [True, False])
@pytest.mark.parametrize("rank", [1, 2, 10])
@pytest.mark.parametrize("randomized", [True, False])
def test_fit(rank, scale, randomized):
    pod = PODBlock(rank, scale)
    pod.fit(toy_snapshots, randomized)
    n_snap = toy_snapshots.shape[0]
    dof = toy_snapshots.shape[1]
    assert pod.basis.shape == (rank, dof)
    assert pod._basis.shape == (n_snap, dof)
    assert pod.singular_values.shape == (rank,)
    assert pod._singular_values.shape == (n_snap,)
    if scale is True:
        assert pod._scaler["mean"].shape == (n_snap,)
        assert pod._scaler["std"].shape == (n_snap,)
        assert pod.scaler["mean"].shape == (rank,)
        assert pod.scaler["std"].shape == (rank,)
        assert pod.scaler["mean"].shape[0] == pod.basis.shape[0]
    else:
        assert pod._scaler == None
        assert pod.scaler == None


def test_forward():
    pod = PODBlock(1)
    pod.fit(toy_snapshots)
    c = pod(toy_snapshots)
    assert c.shape[0] == toy_snapshots.shape[0]
    assert c.shape[1] == pod.rank
    torch.testing.assert_close(c.mean(dim=0), torch.zeros(pod.rank))
    torch.testing.assert_close(c.std(dim=0), torch.ones(pod.rank))

    c = pod(toy_snapshots[0])
    assert c.shape[1] == pod.rank
    assert c.shape[0] == 1

    pod = PODBlock(2, False)
    pod.fit(toy_snapshots)
    c = pod(toy_snapshots)
    torch.testing.assert_close(c, (pod.basis @ toy_snapshots.T).T)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(c.mean(dim=0), torch.zeros(pod.rank))
        torch.testing.assert_close(c.std(dim=0), torch.ones(pod.rank))


@pytest.mark.parametrize("scale", [True, False])
@pytest.mark.parametrize("rank", [1, 2, 10])
@pytest.mark.parametrize("randomized", [True, False])
def test_expand(rank, scale, randomized):
    pod = PODBlock(rank, scale)
    pod.fit(toy_snapshots, randomized)
    c = pod(toy_snapshots)
    torch.testing.assert_close(pod.expand(c), toy_snapshots)
    torch.testing.assert_close(pod.expand(c[0]), toy_snapshots[0].unsqueeze(0))


@pytest.mark.parametrize("scale", [True, False])
@pytest.mark.parametrize("rank", [1, 2, 10])
@pytest.mark.parametrize("randomized", [True, False])
def test_reduce_expand(rank, scale, randomized):
    pod = PODBlock(rank, scale)
    pod.fit(toy_snapshots, randomized)
    torch.testing.assert_close(
        pod.expand(pod.reduce(toy_snapshots)), toy_snapshots
    )
    torch.testing.assert_close(
        pod.expand(pod.reduce(toy_snapshots[0])), toy_snapshots[0].unsqueeze(0)
    )
    # torch.testing.assert_close(pod.expand(pod.reduce(c[0])), c[0])
