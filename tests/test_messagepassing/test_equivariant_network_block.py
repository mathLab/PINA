import pytest
import torch
from pina.model.block.message_passing import EnEquivariantNetworkBlock

# Data for testing
x = torch.rand(10, 4)
pos = torch.rand(10, 3)
velocity = torch.rand(10, 3)
edge_idx = torch.randint(0, 10, (2, 20))
edge_attributes = torch.randn(20, 2)


@pytest.mark.parametrize("node_feature_dim", [1, 3])
@pytest.mark.parametrize("edge_feature_dim", [0, 2])
@pytest.mark.parametrize("pos_dim", [2, 3])
@pytest.mark.parametrize("use_velocity", [True, False])
def test_constructor(node_feature_dim, edge_feature_dim, pos_dim, use_velocity):

    EnEquivariantNetworkBlock(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        pos_dim=pos_dim,
        use_velocity=use_velocity,
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
            use_velocity=use_velocity,
        )

    # Should fail if edge_feature_dim is negative
    with pytest.raises(AssertionError):
        EnEquivariantNetworkBlock(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=-1,
            pos_dim=pos_dim,
            use_velocity=use_velocity,
        )

    # Should fail if pos_dim is negative
    with pytest.raises(AssertionError):
        EnEquivariantNetworkBlock(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            pos_dim=-1,
            use_velocity=use_velocity,
        )

    # Should fail if hidden_dim is negative
    with pytest.raises(AssertionError):
        EnEquivariantNetworkBlock(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            pos_dim=pos_dim,
            hidden_dim=-1,
            use_velocity=use_velocity,
        )

    # Should fail if n_message_layers is negative
    with pytest.raises(AssertionError):
        EnEquivariantNetworkBlock(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            pos_dim=pos_dim,
            n_message_layers=-1,
            use_velocity=use_velocity,
        )

    # Should fail if n_update_layers is negative
    with pytest.raises(AssertionError):
        EnEquivariantNetworkBlock(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            pos_dim=pos_dim,
            n_update_layers=-1,
            use_velocity=use_velocity,
        )

    # Should fail if use_velocity is not boolean
    with pytest.raises(ValueError):
        EnEquivariantNetworkBlock(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            pos_dim=pos_dim,
            use_velocity="False",
        )


@pytest.mark.parametrize("edge_feature_dim", [0, 2])
@pytest.mark.parametrize("use_velocity", [True, False])
def test_forward(edge_feature_dim, use_velocity):

    model = EnEquivariantNetworkBlock(
        node_feature_dim=x.shape[1],
        edge_feature_dim=edge_feature_dim,
        pos_dim=pos.shape[1],
        hidden_dim=64,
        n_message_layers=2,
        n_update_layers=2,
        use_velocity=use_velocity,
    )

    # Manage inputs
    vel = velocity if use_velocity else None
    edge_attr = edge_attributes if edge_feature_dim > 0 else None

    # Checks on output shapes
    output_ = model(
        x=x, pos=pos, edge_index=edge_idx, edge_attr=edge_attr, vel=vel
    )
    assert output_[0].shape == x.shape
    assert output_[1].shape == pos.shape
    if vel is not None:
        assert output_[2].shape == vel.shape


@pytest.mark.parametrize("edge_feature_dim", [0, 2])
@pytest.mark.parametrize("use_velocity", [True, False])
def test_backward(edge_feature_dim, use_velocity):

    model = EnEquivariantNetworkBlock(
        node_feature_dim=x.shape[1],
        edge_feature_dim=edge_feature_dim,
        pos_dim=pos.shape[1],
        hidden_dim=64,
        n_message_layers=2,
        n_update_layers=2,
        use_velocity=use_velocity,
    )

    # Manage inputs
    vel = velocity.requires_grad_() if use_velocity else None
    edge_attr = (
        edge_attributes.requires_grad_() if edge_feature_dim > 0 else None
    )

    if edge_feature_dim == 0:
        output_ = model(
            edge_index=edge_idx,
            x=x.requires_grad_(),
            pos=pos.requires_grad_(),
            vel=vel,
        )
    else:
        output_ = model(
            edge_index=edge_idx,
            x=x.requires_grad_(),
            pos=pos.requires_grad_(),
            edge_attr=edge_attr,
            vel=vel,
        )

    # Checks on gradients
    loss = sum(torch.mean(output_[i]) for i in range(len(output_)))
    loss.backward()
    assert x.grad.shape == x.shape
    assert pos.grad.shape == pos.shape
    if use_velocity:
        assert vel.grad.shape == vel.shape


@pytest.mark.parametrize("edge_feature_dim", [0, 2])
@pytest.mark.parametrize("use_velocity", [True, False])
def test_equivariance(edge_feature_dim, use_velocity):

    # Random rotation
    rotation = torch.linalg.qr(torch.rand(pos.shape[-1], pos.shape[-1])).Q
    if torch.det(rotation) < 0:
        rotation[:, 0] *= -1

    # Random translation
    translation = torch.rand(1, pos.shape[-1])

    model = EnEquivariantNetworkBlock(
        node_feature_dim=x.shape[1],
        edge_feature_dim=edge_feature_dim,
        pos_dim=pos.shape[1],
        hidden_dim=64,
        n_message_layers=2,
        n_update_layers=2,
        use_velocity=use_velocity,
    ).eval()

    # Manage inputs
    vel = velocity if use_velocity else None
    edge_attr = edge_attributes if edge_feature_dim > 0 else None

    # Transform inputs (no translation for velocity)
    pos_rot = pos @ rotation.T + translation
    vel_rot = vel @ rotation.T if use_velocity else vel

    # Get model outputs
    out1 = model(
        x=x, pos=pos, edge_index=edge_idx, edge_attr=edge_attr, vel=vel
    )
    out2 = model(
        x=x, pos=pos_rot, edge_index=edge_idx, edge_attr=edge_attr, vel=vel_rot
    )

    # Unpack outputs
    h1, pos1, *other1 = out1
    h2, pos2, *other2 = out2
    if use_velocity:
        vel1, vel2 = other1[0], other2[0]

    assert torch.allclose(pos2, pos1 @ rotation.T + translation, atol=1e-5)
    assert torch.allclose(h1, h2, atol=1e-5)
    if vel is not None:
        assert torch.allclose(vel2, vel1 @ rotation.T, atol=1e-5)
