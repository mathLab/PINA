import torch
from pina.model import AveragingNeuralOperator
from pina import LabelTensor

output_numb_fields = 5
batch_size = 15


def test_constructor():
    input_numb_fields = 1
    output_numb_fields = 1
    #minimuum constructor
    AveragingNeuralOperator(input_numb_fields,
         output_numb_fields,
         coordinates_indices=['p'],
         field_indices=['v'])

    #all constructor
    AveragingNeuralOperator(input_numb_fields,
         output_numb_fields,
         inner_size=5,
         n_layers=5,
         func=torch.nn.ReLU,
         coordinates_indices=['p'],
         field_indices=['v'])


def test_forward():
    input_numb_fields = 1
    output_numb_fields = 1
    dimension = 1
    input_ = LabelTensor(
        torch.rand(batch_size, 1000, input_numb_fields + dimension), ['p', 'v'])
    ano = AveragingNeuralOperator(input_numb_fields,
               output_numb_fields,
               dimension=dimension,
               coordinates_indices=['p'],
               field_indices=['v'])
    out = ano(input_)
    assert out.shape == torch.Size(
        [batch_size, input_.shape[1], output_numb_fields])


def test_backward():
    input_numb_fields = 1
    dimension = 1
    output_numb_fields = 1
    input_ = LabelTensor(
        torch.rand(batch_size, 1000, dimension + input_numb_fields), 
        ['p', 'v'])
    input_ = input_.requires_grad_()
    avno = AveragingNeuralOperator(input_numb_fields,
                output_numb_fields,
                dimension=dimension,
                coordinates_indices=['p'],
                field_indices=['v'])
    out = avno(input_)
    tmp = torch.linalg.norm(out)
    tmp.backward()
    grad = input_.grad
    assert grad.shape == torch.Size(
        [batch_size, input_.shape[1], dimension + input_numb_fields])
