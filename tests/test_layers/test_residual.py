from pina.model.layers import ResidualBlock, EnhancedLinear
import torch
import torch.nn as nn


def test_constructor_residual_block():

    res_block = ResidualBlock(input_dim=10, output_dim=3, hidden_dim=4)

    res_block = ResidualBlock(input_dim=10,
                              output_dim=3,
                              hidden_dim=4,
                              spectral_norm=True)


def test_forward_residual_block():

    res_block = ResidualBlock(input_dim=10, output_dim=3, hidden_dim=4)

    x = torch.rand(size=(80, 10))
    y = res_block(x)
    assert y.shape[1] == 3
    assert y.shape[0] == x.shape[0]

def test_backward_residual_block():
    
    res_block = ResidualBlock(input_dim=10, output_dim=3, hidden_dim=4)

    x = torch.rand(size=(80, 10))
    x.requires_grad = True
    y = res_block(x)
    l = torch.mean(y)
    l.backward()
    assert x._grad.shape == torch.Size([80,10])

def test_constructor_no_activation_no_dropout():
    linear_layer = nn.Linear(10, 20)
    enhanced_linear = EnhancedLinear(linear_layer)

    assert len(list(enhanced_linear.parameters())) == len(list(linear_layer.parameters()))

def test_constructor_with_activation_no_dropout():
    linear_layer = nn.Linear(10, 20)
    activation = nn.ReLU()
    enhanced_linear = EnhancedLinear(linear_layer, activation)

    assert len(list(enhanced_linear.parameters())) == len(list(linear_layer.parameters())) + len(list(activation.parameters()))

def test_constructor_no_activation_with_dropout():
    linear_layer = nn.Linear(10, 20)
    dropout_prob = 0.5
    enhanced_linear = EnhancedLinear(linear_layer, dropout=dropout_prob)

    assert len(list(enhanced_linear.parameters())) == len(list(linear_layer.parameters()))

def test_constructor_with_activation_with_dropout():
    linear_layer = nn.Linear(10, 20)
    activation = nn.ReLU()
    dropout_prob = 0.5
    enhanced_linear = EnhancedLinear(linear_layer, activation, dropout_prob)

    assert len(list(enhanced_linear.parameters())) == len(list(linear_layer.parameters())) + len(list(activation.parameters()))

def test_forward_enhanced_linear_no_dropout():

    enhanced_linear = EnhancedLinear(nn.Linear(10, 3))

    x = torch.rand(size=(80, 10))
    y = enhanced_linear(x)
    assert y.shape[1] == 3
    assert y.shape[0] == x.shape[0]

def test_backward_enhanced_linear_no_dropout():
    
    enhanced_linear = EnhancedLinear(nn.Linear(10, 3))

    x = torch.rand(size=(80, 10))
    x.requires_grad = True
    y = enhanced_linear(x)
    l = torch.mean(y)
    l.backward()
    assert x._grad.shape == torch.Size([80, 10])

def test_forward_enhanced_linear_dropout():

    enhanced_linear = EnhancedLinear(nn.Linear(10, 3), dropout=0.5)

    x = torch.rand(size=(80, 10))
    y = enhanced_linear(x)
    assert y.shape[1] == 3
    assert y.shape[0] == x.shape[0]

def test_backward_enhanced_linear_dropout():
    
    enhanced_linear = EnhancedLinear(nn.Linear(10, 3), dropout=0.5)

    x = torch.rand(size=(80, 10))
    x.requires_grad = True
    y = enhanced_linear(x)
    l = torch.mean(y)
    l.backward()
    assert x._grad.shape == torch.Size([80, 10])
