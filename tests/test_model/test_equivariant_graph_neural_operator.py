import pytest
import torch
import copy
from pina.model import EquivariantGraphNeuralOperator
from pina.graph import Graph


# Utility to create graphs
def make_graph(include_vel=True, use_edge_attr=True):
    data = dict(
        x=torch.rand(10, 4),
        pos=torch.rand(10, 3),
        edge_index=torch.randint(0, 10, (2, 20)),
        edge_attr=torch.randn(20, 2) if use_edge_attr else None,
    )
    if include_vel:
        data["vel"] = torch.rand(10, 3)
    return Graph(**data)


@pytest.mark.parametrize("n_egno_layers", [1, 3])
@pytest.mark.parametrize("time_steps", [1, 3])
@pytest.mark.parametrize("time_emb_dim", [4, 8])
@pytest.mark.parametrize("max_time_idx", [10, 20])
def test_constructor(n_egno_layers, time_steps, time_emb_dim, max_time_idx):

    # Create graph and model
    graph = make_graph()
    EquivariantGraphNeuralOperator(
        n_egno_layers=n_egno_layers,
        node_feature_dim=graph.x.shape[1],
        edge_feature_dim=graph.edge_attr.shape[1],
        pos_dim=graph.pos.shape[1],
        modes=5,
        time_steps=time_steps,
        time_emb_dim=time_emb_dim,
        max_time_idx=max_time_idx,
    )

    # Should fail if n_egno_layers is negative
    with pytest.raises(AssertionError):
        EquivariantGraphNeuralOperator(
            n_egno_layers=-1,
            node_feature_dim=graph.x.shape[1],
            edge_feature_dim=graph.edge_attr.shape[1],
            pos_dim=graph.pos.shape[1],
            modes=5,
            time_steps=time_steps,
            time_emb_dim=time_emb_dim,
            max_time_idx=max_time_idx,
        )

    # Should fail if time_steps is negative
    with pytest.raises(AssertionError):
        EquivariantGraphNeuralOperator(
            n_egno_layers=n_egno_layers,
            node_feature_dim=graph.x.shape[1],
            edge_feature_dim=graph.edge_attr.shape[1],
            pos_dim=graph.pos.shape[1],
            modes=5,
            time_steps=-1,
            time_emb_dim=time_emb_dim,
            max_time_idx=max_time_idx,
        )

    # Should fail if max_time_idx is negative
    with pytest.raises(AssertionError):
        EquivariantGraphNeuralOperator(
            n_egno_layers=n_egno_layers,
            node_feature_dim=graph.x.shape[1],
            edge_feature_dim=graph.edge_attr.shape[1],
            pos_dim=graph.pos.shape[1],
            modes=5,
            time_steps=time_steps,
            time_emb_dim=time_emb_dim,
            max_time_idx=-1,
        )

    # Should fail if time_emb_dim is negative
    with pytest.raises(AssertionError):
        EquivariantGraphNeuralOperator(
            n_egno_layers=n_egno_layers,
            node_feature_dim=graph.x.shape[1],
            edge_feature_dim=graph.edge_attr.shape[1],
            pos_dim=graph.pos.shape[1],
            modes=5,
            time_steps=time_steps,
            time_emb_dim=-1,
            max_time_idx=max_time_idx,
        )


@pytest.mark.parametrize("n_egno_layers", [1, 3])
@pytest.mark.parametrize("time_steps", [1, 5])
@pytest.mark.parametrize("modes", [1, 3, 10])
@pytest.mark.parametrize("use_edge_attr", [True, False])
def test_forward(n_egno_layers, time_steps, modes, use_edge_attr):

    # Create graph and model
    graph = make_graph(use_edge_attr=use_edge_attr)
    model = EquivariantGraphNeuralOperator(
        n_egno_layers=n_egno_layers,
        node_feature_dim=graph.x.shape[1],
        edge_feature_dim=graph.edge_attr.shape[1] if use_edge_attr else 0,
        pos_dim=graph.pos.shape[1],
        modes=modes,
        time_steps=time_steps,
    )

    # Checks on output shapes
    output_ = model(graph)
    assert output_.x.shape == (time_steps, *graph.x.shape)
    assert output_.pos.shape == (time_steps, *graph.pos.shape)
    assert output_.vel.shape == (time_steps, *graph.vel.shape)

    # Should fail graph has no vel attribute
    with pytest.raises(ValueError):
        graph_no_vel = make_graph(include_vel=False)
        model(graph_no_vel)


@pytest.mark.parametrize("n_egno_layers", [1, 3])
@pytest.mark.parametrize("time_steps", [1, 5])
@pytest.mark.parametrize("modes", [1, 3, 10])
@pytest.mark.parametrize("use_edge_attr", [True, False])
def test_backward(n_egno_layers, time_steps, modes, use_edge_attr):

    # Create graph and model
    graph = make_graph(use_edge_attr=use_edge_attr)
    model = EquivariantGraphNeuralOperator(
        n_egno_layers=n_egno_layers,
        node_feature_dim=graph.x.shape[1],
        edge_feature_dim=graph.edge_attr.shape[1] if use_edge_attr else 0,
        pos_dim=graph.pos.shape[1],
        modes=modes,
        time_steps=time_steps,
    )

    # Set requires_grad and perform forward pass
    graph.x.requires_grad_()
    graph.pos.requires_grad_()
    graph.vel.requires_grad_()
    out = model(graph)

    # Checks on gradients
    loss = torch.mean(out.x) + torch.mean(out.pos) + torch.mean(out.vel)
    loss.backward()
    assert graph.x.grad.shape == graph.x.shape
    assert graph.pos.grad.shape == graph.pos.shape
    assert graph.vel.grad.shape == graph.vel.shape


@pytest.mark.parametrize("n_egno_layers", [1, 3])
@pytest.mark.parametrize("time_steps", [1, 5])
@pytest.mark.parametrize("modes", [1, 3, 10])
@pytest.mark.parametrize("use_edge_attr", [True, False])
def test_equivariance(n_egno_layers, time_steps, modes, use_edge_attr):

    graph = make_graph(use_edge_attr=use_edge_attr)
    model = EquivariantGraphNeuralOperator(
        n_egno_layers=n_egno_layers,
        node_feature_dim=graph.x.shape[1],
        edge_feature_dim=graph.edge_attr.shape[1] if use_edge_attr else 0,
        pos_dim=graph.pos.shape[1],
        modes=modes,
        time_steps=time_steps,
    ).eval()

    # Random rotation
    rotation = torch.linalg.qr(
        torch.rand(graph.pos.shape[1], graph.pos.shape[1])
    ).Q
    if torch.det(rotation) < 0:
        rotation[:, 0] *= -1

    # Random translation
    translation = torch.rand(1, graph.pos.shape[1])

    # Transform graph (no translation for velocity)
    graph_rot = copy.deepcopy(graph)
    graph_rot.pos = graph.pos @ rotation.T + translation
    graph_rot.vel = graph.vel @ rotation.T

    # Get model outputs
    out1 = model(graph)
    out2 = model(graph_rot)

    # Unpack outputs
    h1, pos1, vel1 = out1.x, out1.pos, out1.vel
    h2, pos2, vel2 = out2.x, out2.pos, out2.vel

    assert torch.allclose(pos2, pos1 @ rotation.T + translation, atol=1e-5)
    assert torch.allclose(vel2, vel1 @ rotation.T, atol=1e-5)
    assert torch.allclose(h1, h2, atol=1e-5)
