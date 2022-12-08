import torch

from pina.utils import merge_tensors
from pina.label_tensor import LabelTensor

def test_merge_tensors():
    tensor1 = LabelTensor(torch.rand((20, 3)), ['a', 'b', 'c'])
    tensor2 = LabelTensor(torch.zeros((20, 3)), ['d', 'e', 'f'])
    tensor3 = LabelTensor(torch.ones((30, 3)), ['g', 'h', 'i'])

    merged_tensor = merge_tensors((tensor1, tensor2, tensor3))
    assert tuple(merged_tensor.labels) == ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i')
    assert merged_tensor.shape == (20*20*30, 9)
    assert torch.all(merged_tensor.extract(('d', 'e', 'f')) == 0)
    assert torch.all(merged_tensor.extract(('g', 'h', 'i')) == 1)
