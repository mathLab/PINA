import pytest
import torch
from pina.model.block.message_passing import DeepTensorNetworkBlock

# Data for testing
x = torch.rand(10, 3)
edge_index = torch.randint(0, 10, (2, 20))
edge_attr = torch.randn(20, 2)


@pytest.mark.parametrize("node_feature_dim", [1, 3])
@pytest.mark.parametrize("edge_feature_dim", [3, 5])
def test_constructor(node_feature_dim, edge_feature_dim):

    DeepTensorNetworkBlock(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
    )

    # Should fail if node_feature_dim is negative
    with pytest.raises(AssertionError):
        DeepTensorNetworkBlock(
            node_feature_dim=-1, edge_feature_dim=edge_feature_dim
        )

    # Should fail if edge_feature_dim is negative
    with pytest.raises(AssertionError):
        DeepTensorNetworkBlock(
            node_feature_dim=node_feature_dim, edge_feature_dim=-1
        )


def test_forward():

    model = DeepTensorNetworkBlock(
        node_feature_dim=x.shape[1],
        edge_feature_dim=edge_attr.shape[1],
    )

    output_ = model(edge_index=edge_index, x=x, edge_attr=edge_attr)
    assert output_.shape == x.shape


def test_backward():

    model = DeepTensorNetworkBlock(
        node_feature_dim=x.shape[1],
        edge_feature_dim=edge_attr.shape[1],
    )

    output_ = model(
        edge_index=edge_index,
        x=x.requires_grad_(),
        edge_attr=edge_attr.requires_grad_(),
    )

    loss = torch.mean(output_)
    loss.backward()
    assert x.grad.shape == x.shape
