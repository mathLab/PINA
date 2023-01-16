from convolution_2d import ContinuousConv2D
import torch
from utils import prod


def make_grid(x):
    def _transform_image(image):

        # extracting image info
        channels, dimension = image.size()[0], image.size()[1:]

        # initializing transfomed image
        coordinates = torch.zeros(
            [channels, prod(dimension), len(dimension) + 1]).to(image.device)

        # creating the n dimensional mesh grid
        values_mesh = [torch.arange(0, dim).float().to(
            image.device) for dim in dimension]
        mesh = torch.meshgrid(values_mesh)
        coordinates_mesh = [x.reshape(-1, 1) for x in mesh]
        coordinates_mesh.append(0)

        for count, channel in enumerate(image):
            coordinates_mesh[-1] = channel.reshape(-1, 1)
            coordinates[count] = torch.cat(coordinates_mesh, dim=1)

        return coordinates

    output = [_transform_image(current_image) for current_image in x]
    return torch.stack(output).to(x.device)


class MLP(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self. model = torch.nn.Sequential(torch.nn.Linear(2, 8),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(8, 8),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(8, 1))

    def forward(self, x):
        return self.model(x)


# INPUTS
channel_input = 2
channel_output = 6
batch = 2
N = 10
dim = [3, 3]
stride = {"domain": [10, 10],
          "start": [0, 0],
          "jumps": [3, 3],
          "direction": [1, 1.]}
dim_filter = len(dim)
dim_input = (batch, channel_input, 10, dim_filter)
dim_output = (batch, channel_output, 4, dim_filter)
x = torch.rand(dim_input)
x = make_grid(x)


def test_constructor():
    model = MLP

    conv = ContinuousConv2D(channel_input,
                            channel_output,
                            dim,
                            stride,
                            model=model)
    conv = ContinuousConv2D(channel_input,
                            channel_output,
                            dim,
                            stride,
                            model=None)


def test_forward():
    model = MLP

    conv = ContinuousConv2D(channel_input,
                            channel_output,
                            dim,
                            stride,
                            model=model)
    conv(x)


def test_transpose():
    model = MLP

    conv = ContinuousConv2D(channel_input,
                            channel_output,
                            dim,
                            stride,
                            model=model)

    integrals = conv(x)
    conv.transpose(integrals[..., -1], x)
