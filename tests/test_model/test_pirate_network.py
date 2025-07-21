import torch
import pytest
from pina.model import PirateNet
from pina.model.block import FourierFeatureEmbedding

data = torch.rand((20, 3))


@pytest.mark.parametrize("inner_size", [10, 20])
@pytest.mark.parametrize("n_layers", [1, 3])
@pytest.mark.parametrize("output_dimension", [2, 4])
def test_constructor(inner_size, n_layers, output_dimension):

    # Loop over the default and custom embedding
    for embedding in [None, torch.nn.Linear(data.shape[1], inner_size)]:

        # Constructor
        model = PirateNet(
            input_dimension=data.shape[1],
            inner_size=inner_size,
            output_dimension=output_dimension,
            embedding=embedding,
            n_layers=n_layers,
            activation=torch.nn.Tanh,
        )

        # Check the default embedding
        if embedding is None:
            assert isinstance(model.embedding, FourierFeatureEmbedding)
            assert model.embedding.sigma == 2.0

        # Should fail if input_dimension is negative
        with pytest.raises(AssertionError):
            PirateNet(
                input_dimension=-1,
                inner_size=inner_size,
                output_dimension=output_dimension,
                embedding=embedding,
                n_layers=n_layers,
                activation=torch.nn.Tanh,
            )

        # Should fail if inner_size is negative
        with pytest.raises(AssertionError):
            PirateNet(
                input_dimension=data.shape[1],
                inner_size=-1,
                output_dimension=output_dimension,
                embedding=embedding,
                n_layers=n_layers,
                activation=torch.nn.Tanh,
            )

        # Should fail if output_dimension is negative
        with pytest.raises(AssertionError):
            PirateNet(
                input_dimension=data.shape[1],
                inner_size=inner_size,
                output_dimension=-1,
                embedding=embedding,
                n_layers=n_layers,
                activation=torch.nn.Tanh,
            )

        # Should fail if n_layers is negative
        with pytest.raises(AssertionError):
            PirateNet(
                input_dimension=data.shape[1],
                inner_size=inner_size,
                output_dimension=output_dimension,
                embedding=embedding,
                n_layers=-1,
                activation=torch.nn.Tanh,
            )


@pytest.mark.parametrize("inner_size", [10, 20])
@pytest.mark.parametrize("n_layers", [1, 3])
@pytest.mark.parametrize("output_dimension", [2, 4])
def test_forward(inner_size, n_layers, output_dimension):

    # Loop over the default and custom embedding
    for embedding in [None, torch.nn.Linear(data.shape[1], inner_size)]:

        model = PirateNet(
            input_dimension=data.shape[1],
            inner_size=inner_size,
            output_dimension=output_dimension,
            embedding=embedding,
            n_layers=n_layers,
            activation=torch.nn.Tanh,
        )

        output_ = model(data)
        assert output_.shape == (data.shape[0], output_dimension)


@pytest.mark.parametrize("inner_size", [10, 20])
@pytest.mark.parametrize("n_layers", [1, 3])
@pytest.mark.parametrize("output_dimension", [2, 4])
def test_backward(inner_size, n_layers, output_dimension):

    # Loop over the default and custom embedding
    for embedding in [None, torch.nn.Linear(data.shape[1], inner_size)]:

        model = PirateNet(
            input_dimension=data.shape[1],
            inner_size=inner_size,
            output_dimension=output_dimension,
            embedding=embedding,
            n_layers=n_layers,
            activation=torch.nn.Tanh,
        )

        data.requires_grad_()
        output_ = model(data)

        loss = torch.mean(output_)
        loss.backward()
        assert data.grad.shape == data.shape
