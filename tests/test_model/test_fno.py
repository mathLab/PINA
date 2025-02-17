import torch
from pina.model import FNO

output_channels = 5
batch_size = 4
resolution = [4, 6, 8]
lifting_dim = 24


def test_constructor():
    input_channels = 3
    lifting_net = torch.nn.Linear(input_channels, lifting_dim)
    projecting_net = torch.nn.Linear(60, output_channels)

    # simple constructor
    FNO(lifting_net=lifting_net,
        projecting_net=projecting_net,
        n_modes=5,
        dimensions=3,
        inner_size=60,
        n_layers=5)

    # simple constructor with n_modes list
    FNO(lifting_net=lifting_net,
        projecting_net=projecting_net,
        n_modes=[5, 3, 2],
        dimensions=3,
        inner_size=60,
        n_layers=5)

    # simple constructor with n_modes list of list
    FNO(lifting_net=lifting_net,
        projecting_net=projecting_net,
        n_modes=[[5, 3, 2], [5, 3, 2]],
        dimensions=3,
        inner_size=60,
        n_layers=2)

    # simple constructor with n_modes list of list
    projecting_net = torch.nn.Linear(50, output_channels)
    FNO(lifting_net=lifting_net,
        projecting_net=projecting_net,
        n_modes=5,
        dimensions=3,
        layers=[50, 50])


def test_1d_forward():
    input_channels = 1
    input_ = torch.rand(batch_size, resolution[0], input_channels)
    lifting_net = torch.nn.Linear(input_channels, lifting_dim)
    projecting_net = torch.nn.Linear(60, output_channels)
    fno = FNO(lifting_net=lifting_net,
              projecting_net=projecting_net,
              n_modes=5,
              dimensions=1,
              inner_size=60,
              n_layers=2)
    out = fno(input_)
    assert out.shape == torch.Size([batch_size, resolution[0], output_channels])


def test_1d_backward():
    input_channels = 1
    input_ = torch.rand(batch_size, resolution[0], input_channels)
    lifting_net = torch.nn.Linear(input_channels, lifting_dim)
    projecting_net = torch.nn.Linear(60, output_channels)
    fno = FNO(lifting_net=lifting_net,
              projecting_net=projecting_net,
              n_modes=5,
              dimensions=1,
              inner_size=60,
              n_layers=2)
    input_.requires_grad = True
    out = fno(input_)
    l = torch.mean(out)
    l.backward()
    assert input_.grad.shape == torch.Size([batch_size, resolution[0], input_channels])


def test_2d_forward():
    input_channels = 2
    input_ = torch.rand(batch_size, resolution[0], resolution[1],
                        input_channels)
    lifting_net = torch.nn.Linear(input_channels, lifting_dim)
    projecting_net = torch.nn.Linear(60, output_channels)
    fno = FNO(lifting_net=lifting_net,
              projecting_net=projecting_net,
              n_modes=5,
              dimensions=2,
              inner_size=60,
              n_layers=2)
    out = fno(input_)
    assert out.shape == torch.Size(
        [batch_size, resolution[0], resolution[1], output_channels])


def test_2d_backward():
    input_channels = 2
    input_ = torch.rand(batch_size, resolution[0], resolution[1],
                        input_channels)
    lifting_net = torch.nn.Linear(input_channels, lifting_dim)
    projecting_net = torch.nn.Linear(60, output_channels)
    fno = FNO(lifting_net=lifting_net,
              projecting_net=projecting_net,
              n_modes=5,
              dimensions=2,
              inner_size=60,
              n_layers=2)
    input_.requires_grad = True
    out = fno(input_)
    l = torch.mean(out)
    l.backward()
    assert input_.grad.shape == torch.Size([
        batch_size, resolution[0], resolution[1], input_channels
    ])


def test_3d_forward():
    input_channels = 3
    input_ = torch.rand(batch_size, resolution[0], resolution[1], resolution[2],
                        input_channels)
    lifting_net = torch.nn.Linear(input_channels, lifting_dim)
    projecting_net = torch.nn.Linear(60, output_channels)
    fno = FNO(lifting_net=lifting_net,
              projecting_net=projecting_net,
              n_modes=5,
              dimensions=3,
              inner_size=60,
              n_layers=2)
    out = fno(input_)
    assert out.shape == torch.Size([
        batch_size, resolution[0], resolution[1], resolution[2], output_channels
    ])


def test_3d_backward():
    input_channels = 3
    input_ = torch.rand(batch_size, resolution[0], resolution[1], resolution[2],
                        input_channels)
    lifting_net = torch.nn.Linear(input_channels, lifting_dim)
    projecting_net = torch.nn.Linear(60, output_channels)
    fno = FNO(lifting_net=lifting_net,
              projecting_net=projecting_net,
              n_modes=5,
              dimensions=3,
              inner_size=60,
              n_layers=2)
    input_.requires_grad = True
    out = fno(input_)
    l = torch.mean(out)
    l.backward()
    assert input_.grad.shape == torch.Size([
        batch_size, resolution[0], resolution[1], resolution[2], input_channels
    ])
