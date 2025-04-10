import torch
import pytest
from pina.data.dataset import PinaDatasetFactory, PinaGraphDataset
from pina.graph import KNNGraph
from torch_geometric.data import Data

x = torch.rand((100, 20, 10))
pos = torch.rand((100, 20, 2))
input_ = [
    KNNGraph(x=x_, pos=pos_, neighbours=3, edge_attr=True)
    for x_, pos_ in zip(x, pos)
]
output_ = torch.rand((100, 20, 10))

x_2 = torch.rand((50, 20, 10))
pos_2 = torch.rand((50, 20, 2))
input_2_ = [
    KNNGraph(x=x_, pos=pos_, neighbours=3, edge_attr=True)
    for x_, pos_ in zip(x_2, pos_2)
]
output_2_ = torch.rand((50, 20, 10))


# Problem with a single condition
conditions_dict_single = {
    "data": {
        "input": input_,
        "target": output_,
    }
}
max_conditions_lengths_single = {"data": 100}

# Problem with multiple conditions
conditions_dict_multi = {
    "data_1": {
        "input": input_,
        "target": output_,
    },
    "data_2": {
        "input": input_2_,
        "target": output_2_,
    },
}

max_conditions_lengths_multi = {"data_1": 100, "data_2": 50}


@pytest.mark.parametrize(
    "conditions_dict, max_conditions_lengths",
    [
        (conditions_dict_single, max_conditions_lengths_single),
        (conditions_dict_multi, max_conditions_lengths_multi),
    ],
)
def test_constructor(conditions_dict, max_conditions_lengths):
    dataset = PinaDatasetFactory(
        conditions_dict,
        max_conditions_lengths=max_conditions_lengths,
        automatic_batching=True,
    )
    assert isinstance(dataset, PinaGraphDataset)
    assert len(dataset) == 100


@pytest.mark.parametrize(
    "conditions_dict, max_conditions_lengths",
    [
        (conditions_dict_single, max_conditions_lengths_single),
        (conditions_dict_multi, max_conditions_lengths_multi),
    ],
)
def test_getitem(conditions_dict, max_conditions_lengths):
    dataset = PinaDatasetFactory(
        conditions_dict,
        max_conditions_lengths=max_conditions_lengths,
        automatic_batching=True,
    )
    data = dataset[50]
    assert isinstance(data, dict)
    assert all([isinstance(d["input"], Data) for d in data.values()])
    assert all([isinstance(d["target"], torch.Tensor) for d in data.values()])
    assert all(
        [d["input"].x.shape == torch.Size((20, 10)) for d in data.values()]
    )
    assert all(
        [d["target"].shape == torch.Size((20, 10)) for d in data.values()]
    )
    assert all(
        [
            d["input"].edge_index.shape == torch.Size((2, 60))
            for d in data.values()
        ]
    )
    assert all([d["input"].edge_attr.shape[0] == 60 for d in data.values()])

    data = dataset.fetch_from_idx_list([i for i in range(20)])
    assert isinstance(data, dict)
    assert all([isinstance(d["input"], Data) for d in data.values()])
    assert all([isinstance(d["target"], torch.Tensor) for d in data.values()])
    assert all(
        [d["input"].x.shape == torch.Size((400, 10)) for d in data.values()]
    )
    assert all(
        [d["target"].shape == torch.Size((400, 10)) for d in data.values()]
    )
    assert all(
        [
            d["input"].edge_index.shape == torch.Size((2, 1200))
            for d in data.values()
        ]
    )
    assert all([d["input"].edge_attr.shape[0] == 1200 for d in data.values()])


def test_input_single_condition():
    dataset = PinaDatasetFactory(
        conditions_dict_single,
        max_conditions_lengths=max_conditions_lengths_single,
        automatic_batching=True,
    )
    input_ = dataset.input
    assert isinstance(input_, dict)
    assert isinstance(input_["data"], list)
    assert all([isinstance(d, Data) for d in input_["data"]])


def test_input_multi_condition():
    dataset = PinaDatasetFactory(
        conditions_dict_multi,
        max_conditions_lengths=max_conditions_lengths_multi,
        automatic_batching=True,
    )
    input_ = dataset.input
    assert isinstance(input_, dict)
    assert isinstance(input_["data_1"], list)
    assert all([isinstance(d, Data) for d in input_["data_1"]])
    assert isinstance(input_["data_2"], list)
    assert all([isinstance(d, Data) for d in input_["data_2"]])
