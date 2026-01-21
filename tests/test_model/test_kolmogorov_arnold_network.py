import torch
import pytest

from pina.model import KolmogorovArnoldNetwork

data = torch.rand((20, 3))
input_vars = 3
output_vars = 1


def test_constructor():
    KolmogorovArnoldNetwork([input_vars, output_vars])
    KolmogorovArnoldNetwork([input_vars, 10, 20, output_vars])
    KolmogorovArnoldNetwork(
        [input_vars, 10, 20, output_vars],
        k=3,
        num=5
    )
    KolmogorovArnoldNetwork(
        [input_vars, 10, 20, output_vars],
        k=3,
        num=5,
        grid_eps=0.05,
        grid_range=[-2, 2]
    )
    KolmogorovArnoldNetwork(
        [input_vars, 10, output_vars],
        base_function=torch.nn.Tanh(),
        scale_sp=0.5,
        sparse_init=True
    )


def test_constructor_wrong():
    with pytest.raises(ValueError):
        KolmogorovArnoldNetwork([input_vars])
    with pytest.raises(ValueError):
        KolmogorovArnoldNetwork([])


def test_forward():
    dim_in, dim_out = 3, 2
    kan = KolmogorovArnoldNetwork([dim_in, dim_out])
    output_ = kan(data)
    assert output_.shape == (data.shape[0], dim_out)


def test_forward_multilayer():
    dim_in, dim_out = 3, 2
    kan = KolmogorovArnoldNetwork([dim_in, 10, 5, dim_out])
    output_ = kan(data)
    assert output_.shape == (data.shape[0], dim_out)


def test_backward():
    dim_in, dim_out = 3, 2
    kan = KolmogorovArnoldNetwork([dim_in, dim_out])
    data.requires_grad = True
    output_ = kan(data)
    loss = torch.mean(output_)
    loss.backward()
    assert data._grad.shape == torch.Size([20, 3])


def test_get_num_parameters():
    kan = KolmogorovArnoldNetwork([3, 5, 2])
    num_params = kan.get_num_parameters()
    assert num_params > 0
    assert isinstance(num_params, int)

from pina.problem.zoo import Poisson2DSquareProblem
from pina.solver import PINN
from pina.trainer import Trainer

def test_train_poisson():
    problem = Poisson2DSquareProblem()
    problem.discretise_domain(n=10, mode="random", domains="all")

    model = KolmogorovArnoldNetwork([2, 3, 1], k=3, num=5)
    solver = PINN(model=model, problem=problem)
    trainer = Trainer(
        solver=solver,
        max_epochs=10,
        accelerator="cpu",
        batch_size=100,
        train_size=1.0,
        val_size=0.0,
        test_size=0.0,
    )
    trainer.train()



# def test_update_grid_from_samples():
#     kan = KolmogorovArnoldNetwork([3, 5, 2])
#     samples = torch.randn(50, 3)
#     kan.update_grid_from_samples(samples, mode='sample')
#     # Check that the network still works after grid update
#     output = kan(data)
#     assert output.shape == (data.shape[0], 2)


# def test_update_grid_resolution():
#     kan = KolmogorovArnoldNetwork([3, 5, 2], num=3)
#     kan.update_grid_resolution(5)
#     # Check that the network still works after resolution update
#     output = kan(data)
#     assert output.shape == (data.shape[0], 2)


# def test_enable_sparsification():
#     kan = KolmogorovArnoldNetwork([3, 5, 2])
#     kan.enable_sparsification(threshold=1e-4)
#     # Check that the network still works after sparsification
#     output = kan(data)
#     assert output.shape == (data.shape[0], 2)


# def test_get_activation_statistics():
#     kan = KolmogorovArnoldNetwork([3, 5, 2])
#     stats = kan.get_activation_statistics(data)
#     assert isinstance(stats, dict)
#     assert 'layer_0' in stats
#     assert 'layer_1' in stats
#     assert 'mean' in stats['layer_0']
#     assert 'std' in stats['layer_0']
#     assert 'min' in stats['layer_0']
#     assert 'max' in stats['layer_0']


# def test_get_network_grid_statistics():
#     kan = KolmogorovArnoldNetwork([3, 5, 2])
#     stats = kan.get_network_grid_statistics()
#     assert isinstance(stats, dict)
#     assert 'layer_0' in stats
#     assert 'layer_1' in stats


# def test_save_act():
#     kan = KolmogorovArnoldNetwork([3, 5, 2], save_act=True)
#     output = kan(data)
#     assert hasattr(kan, 'acts')
#     assert len(kan.acts) == 3  # input + 2 layers
#     assert kan.acts[0].shape == data.shape
#     assert kan.acts[-1].shape == output.shape


# def test_save_act_disabled():
#     kan = KolmogorovArnoldNetwork([3, 5, 2], save_act=False)
#     _ = kan(data)
#     assert hasattr(kan, 'acts')
#     # Only the first activation (input) is saved
#     assert len(kan.acts) == 1
