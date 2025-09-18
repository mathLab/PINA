import pytest
import torch
from pina.model.block.message_passing import EquivariantGraphNeuralOperatorBlock

# Data for testing. Shapes: (time, nodes, features)
x = torch.rand(5, 10, 4)
pos = torch.rand(5, 10, 3)
vel = torch.rand(5, 10, 3)

# Edge index and attributes
edge_idx = torch.randint(0, 10, (2, 20))
edge_attributes = torch.randn(20, 2)


@pytest.mark.parametrize("node_feature_dim", [1, 3])
@pytest.mark.parametrize("edge_feature_dim", [0, 2])
@pytest.mark.parametrize("pos_dim", [2, 3])
@pytest.mark.parametrize("modes", [1, 5])
def test_constructor(node_feature_dim, edge_feature_dim, pos_dim, modes):

    EquivariantGraphNeuralOperatorBlock(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        pos_dim=pos_dim,
        modes=modes,
    )

    # Should fail if modes is negative
    with pytest.raises(AssertionError):
        EquivariantGraphNeuralOperatorBlock(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            pos_dim=pos_dim,
            modes=-1,
        )


@pytest.mark.parametrize("modes", [1, 5])
def test_forward(modes):

    model = EquivariantGraphNeuralOperatorBlock(
        node_feature_dim=x.shape[2],
        edge_feature_dim=edge_attributes.shape[1],
        pos_dim=pos.shape[2],
        modes=modes,
    )

    output_ = model(
        x=x,
        pos=pos,
        vel=vel,
        edge_index=edge_idx,
        edge_attr=edge_attributes,
    )

    # Checks on output shapes
    assert output_[0].shape == x.shape
    assert output_[1].shape == pos.shape
    assert output_[2].shape == vel.shape


@pytest.mark.parametrize("modes", [1, 5])
def test_backward(modes):

    model = EquivariantGraphNeuralOperatorBlock(
        node_feature_dim=x.shape[2],
        edge_feature_dim=edge_attributes.shape[1],
        pos_dim=pos.shape[2],
        modes=modes,
    )

    output_ = model(
        x=x.requires_grad_(),
        pos=pos.requires_grad_(),
        vel=vel.requires_grad_(),
        edge_index=edge_idx,
        edge_attr=edge_attributes.requires_grad_(),
    )

    # Checks on gradients
    loss = sum(torch.mean(output_[i]) for i in range(len(output_)))
    loss.backward()
    assert x.grad.shape == x.shape
    assert pos.grad.shape == pos.shape
    assert vel.grad.shape == vel.shape


@pytest.mark.parametrize("modes", [1, 5])
def test_equivariance(modes):

    # Random rotation
    rotation = torch.linalg.qr(torch.rand(pos.shape[2], pos.shape[2])).Q
    if torch.det(rotation) < 0:
        rotation[:, 0] *= -1

    # Random translation
    translation = torch.rand(1, pos.shape[2])

    model = EquivariantGraphNeuralOperatorBlock(
        node_feature_dim=x.shape[2],
        edge_feature_dim=edge_attributes.shape[1],
        pos_dim=pos.shape[2],
        modes=modes,
    ).eval()

    # Transform inputs (no translation for velocity)
    pos_rot = pos @ rotation.T + translation
    vel_rot = vel @ rotation.T

    # Get model outputs
    out1 = model(
        x=x,
        pos=pos,
        vel=vel,
        edge_index=edge_idx,
        edge_attr=edge_attributes,
    )
    out2 = model(
        x=x,
        pos=pos_rot,
        vel=vel_rot,
        edge_index=edge_idx,
        edge_attr=edge_attributes,
    )

    # Unpack outputs
    h1, pos1, vel1 = out1
    h2, pos2, vel2 = out2

    assert torch.allclose(pos2, pos1 @ rotation.T + translation, atol=1e-5)
    assert torch.allclose(vel2, vel1 @ rotation.T, atol=1e-5)
    assert torch.allclose(h1, h2, atol=1e-5)
