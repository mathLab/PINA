import pytest
import torch
from pina.model.block.message_passing import EnEquivariantNetworkBlock

# Data for testing
x = torch.rand(10, 4)
pos = torch.rand(10, 3)
edge_index = torch.randint(0, 10, (2, 20))
edge_attr = torch.randn(20, 2)


@pytest.mark.parametrize("node_feature_dim", [1, 3])
@pytest.mark.parametrize("edge_feature_dim", [0, 2])
@pytest.mark.parametrize("pos_dim", [2, 3])
def test_constructor(node_feature_dim, edge_feature_dim, pos_dim):

    EnEquivariantNetworkBlock(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        pos_dim=pos_dim,
        hidden_dim=64,
        n_message_layers=2,
        n_update_layers=2,
    )

    # Should fail if node_feature_dim is negative
    with pytest.raises(AssertionError):
        EnEquivariantNetworkBlock(
            node_feature_dim=-1,
            edge_feature_dim=edge_feature_dim,
            pos_dim=pos_dim,
        )

    # Should fail if edge_feature_dim is negative
    with pytest.raises(AssertionError):
        EnEquivariantNetworkBlock(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=-1,
            pos_dim=pos_dim,
        )

    # Should fail if pos_dim is negative
    with pytest.raises(AssertionError):
        EnEquivariantNetworkBlock(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            pos_dim=-1,
        )

    # Should fail if hidden_dim is negative
    with pytest.raises(AssertionError):
        EnEquivariantNetworkBlock(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            pos_dim=pos_dim,
            hidden_dim=-1,
        )

    # Should fail if n_message_layers is negative
    with pytest.raises(AssertionError):
        EnEquivariantNetworkBlock(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            pos_dim=pos_dim,
            n_message_layers=-1,
        )

    # Should fail if n_update_layers is negative
    with pytest.raises(AssertionError):
        EnEquivariantNetworkBlock(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            pos_dim=pos_dim,
            n_update_layers=-1,
        )


@pytest.mark.parametrize("edge_feature_dim", [0, 2])
def test_forward(edge_feature_dim):

    model = EnEquivariantNetworkBlock(
        node_feature_dim=x.shape[1],
        edge_feature_dim=edge_feature_dim,
        pos_dim=pos.shape[1],
        hidden_dim=64,
        n_message_layers=2,
        n_update_layers=2,
    )

    if edge_feature_dim == 0:
        output_ = model(edge_index=edge_index, x=x, pos=pos)
    else:
        output_ = model(
            edge_index=edge_index, x=x, pos=pos, edge_attr=edge_attr
        )

    assert output_[0].shape == x.shape
    assert output_[1].shape == pos.shape


@pytest.mark.parametrize("edge_feature_dim", [0, 2])
def test_backward(edge_feature_dim):

    model = EnEquivariantNetworkBlock(
        node_feature_dim=x.shape[1],
        edge_feature_dim=edge_feature_dim,
        pos_dim=pos.shape[1],
        hidden_dim=64,
        n_message_layers=2,
        n_update_layers=2,
    )

    if edge_feature_dim == 0:
        output_ = model(
            edge_index=edge_index,
            x=x.requires_grad_(),
            pos=pos.requires_grad_(),
        )
    else:
        output_ = model(
            edge_index=edge_index,
            x=x.requires_grad_(),
            pos=pos.requires_grad_(),
            edge_attr=edge_attr.requires_grad_(),
        )

    loss = torch.mean(output_[0])
    loss.backward()
    assert x.grad.shape == x.shape
    assert pos.grad.shape == pos.shape


def test_equivariance():

    # Graph to be fully connected and undirected
    edge_index = torch.combinations(torch.arange(x.shape[0]), r=2).T
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Random rotation (det(rotation) should be 1)
    rotation = torch.linalg.qr(torch.rand(pos.shape[-1], pos.shape[-1])).Q
    if torch.det(rotation) < 0:
        rotation[:, 0] *= -1

    # Random translation
    translation = torch.rand(1, pos.shape[-1])

    model = EnEquivariantNetworkBlock(
        node_feature_dim=x.shape[1],
        edge_feature_dim=0,
        pos_dim=pos.shape[1],
        hidden_dim=64,
        n_message_layers=2,
        n_update_layers=2,
    ).eval()

    h1, pos1 = model(edge_index=edge_index, x=x, pos=pos)
    h2, pos2 = model(
        edge_index=edge_index, x=x, pos=pos @ rotation.T + translation
    )

    # Transform model output
    pos1_transformed = (pos1 @ rotation.T) + translation

    assert torch.allclose(pos2, pos1_transformed, atol=1e-5)
    assert torch.allclose(h1, h2, atol=1e-5)
