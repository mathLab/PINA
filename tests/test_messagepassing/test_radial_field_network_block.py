import pytest
import torch
from pina.model.block.message_passing import RadialFieldNetworkBlock

# Data for testing
x = torch.rand(10, 3)
edge_index = torch.randint(0, 10, (2, 20))


@pytest.mark.parametrize("node_feature_dim", [1, 3])
def test_constructor(node_feature_dim):

    RadialFieldNetworkBlock(
        node_feature_dim=node_feature_dim,
        hidden_dim=64,
        n_layers=2,
    )

    # Should fail if node_feature_dim is negative
    with pytest.raises(AssertionError):
        RadialFieldNetworkBlock(
            node_feature_dim=-1,
            hidden_dim=64,
            n_layers=2,
        )

    # Should fail if hidden_dim is negative
    with pytest.raises(AssertionError):
        RadialFieldNetworkBlock(
            node_feature_dim=node_feature_dim,
            hidden_dim=-1,
            n_layers=2,
        )

    # Should fail if n_layers is negative
    with pytest.raises(AssertionError):
        RadialFieldNetworkBlock(
            node_feature_dim=node_feature_dim,
            hidden_dim=64,
            n_layers=-1,
        )


def test_forward():

    model = RadialFieldNetworkBlock(
        node_feature_dim=x.shape[1],
        hidden_dim=64,
        n_layers=2,
    )

    output_ = model(edge_index=edge_index, x=x)
    assert output_.shape == x.shape


def test_backward():

    model = RadialFieldNetworkBlock(
        node_feature_dim=x.shape[1],
        hidden_dim=64,
        n_layers=2,
    )

    output_ = model(edge_index=edge_index, x=x.requires_grad_())
    loss = torch.mean(output_)
    loss.backward()
    assert x.grad.shape == x.shape


def test_equivariance():

    # Graph to be fully connected and undirected
    edge_index = torch.combinations(torch.arange(x.shape[0]), r=2).T
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Random rotation (det(rotation) should be 1)
    rotation = torch.linalg.qr(torch.rand(x.shape[-1], x.shape[-1])).Q
    if torch.det(rotation) < 0:
        rotation[:, 0] *= -1

    # Random translation
    translation = torch.rand(1, x.shape[-1])

    model = RadialFieldNetworkBlock(node_feature_dim=x.shape[1]).eval()

    pos1 = model(edge_index=edge_index, x=x)
    pos2 = model(edge_index=edge_index, x=x @ rotation.T + translation)

    # Transform model output
    pos1_transformed = (pos1 @ rotation.T) + translation

    assert torch.allclose(pos2, pos1_transformed, atol=1e-5)
