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
