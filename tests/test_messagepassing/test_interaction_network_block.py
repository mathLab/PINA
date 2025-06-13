import pytest
import torch
from pina.model.block.message_passing import InteractionNetworkBlock

# Data for testing
x = torch.rand(10, 3)
edge_index = torch.randint(0, 10, (2, 20))
edge_attr = torch.randn(20, 2)


@pytest.mark.parametrize("node_feature_dim", [1, 3])
@pytest.mark.parametrize("edge_feature_dim", [0, 2])
def test_constructor(node_feature_dim, edge_feature_dim):

    InteractionNetworkBlock(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=64,
        n_message_layers=2,
        n_update_layers=2,
    )

    # Should fail if node_feature_dim is negative
    with pytest.raises(AssertionError):
        InteractionNetworkBlock(node_feature_dim=-1)

    # Should fail if edge_feature_dim is negative
    with pytest.raises(AssertionError):
        InteractionNetworkBlock(node_feature_dim=3, edge_feature_dim=-1)

    # Should fail if hidden_dim is negative
    with pytest.raises(AssertionError):
        InteractionNetworkBlock(node_feature_dim=3, hidden_dim=-1)

    # Should fail if n_message_layers is negative
    with pytest.raises(AssertionError):
        InteractionNetworkBlock(node_feature_dim=3, n_message_layers=-1)

    # Should fail if n_update_layers is negative
    with pytest.raises(AssertionError):
        InteractionNetworkBlock(node_feature_dim=3, n_update_layers=-1)


@pytest.mark.parametrize("edge_feature_dim", [0, 2])
def test_forward(edge_feature_dim):

    model = InteractionNetworkBlock(
        node_feature_dim=x.shape[1],
        edge_feature_dim=edge_feature_dim,
        hidden_dim=64,
        n_message_layers=2,
        n_update_layers=2,
    )

    if edge_feature_dim == 0:
        output_ = model(edge_index=edge_index, x=x)
    else:
        output_ = model(edge_index=edge_index, x=x, edge_attr=edge_attr)
    assert output_.shape == x.shape


@pytest.mark.parametrize("edge_feature_dim", [0, 2])
def test_backward(edge_feature_dim):

    model = InteractionNetworkBlock(
        node_feature_dim=x.shape[1],
        edge_feature_dim=edge_feature_dim,
        hidden_dim=64,
        n_message_layers=2,
        n_update_layers=2,
    )

    if edge_feature_dim == 0:
        output_ = model(edge_index=edge_index, x=x.requires_grad_())
    else:
        output_ = model(
            edge_index=edge_index,
            x=x.requires_grad_(),
            edge_attr=edge_attr.requires_grad_(),
        )

    loss = torch.mean(output_)
    loss.backward()
    assert x.grad.shape == x.shape
