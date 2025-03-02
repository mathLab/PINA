import torch
import pytest
from pina.data.dataset import PinaDatasetFactory, PinaGraphDataset
from pina.graph import KNNGraph
from torch_geometric.data import Data

x = torch.rand((100, 20, 10))
pos = torch.rand((100, 20, 2))
output_ = torch.rand((100, 20, 10))
input_ = [
    KNNGraph(x=x_, pos=pos_, neighbours=3, edge_attr=True, y=y_)
    for x_, pos_, y_ in zip(x, pos, output_)
]

x_2 = torch.rand((50, 20, 10))
pos_2 = torch.rand((50, 20, 2))
output_2_ = torch.rand((50, 20, 10))
input_2_ = [
    KNNGraph(x=x_, pos=pos_, neighbours=3, edge_attr=True, y=y_)
    for x_, pos_, y_ in zip(x_2, pos_2, output_2_)
]


# Problem with a single condition
conditions_dict_single = {
    "data": {
        "graph": input_,
    }
}
max_conditions_lengths_single = {"data": 100}

# Problem with multiple conditions
conditions_dict_single_multi = {
    "data_1": {
        "graph": input_,
    },
    "data_2": {
        "graph": input_2_,
    },
}

max_conditions_lengths_multi = {"data_1": 100, "data_2": 50}


@pytest.mark.parametrize(
    "conditions_dict, max_conditions_lengths",
    [
        (conditions_dict_single, max_conditions_lengths_single),
        (conditions_dict_single_multi, max_conditions_lengths_multi),
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
        (conditions_dict_single_multi, max_conditions_lengths_multi),
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
    assert all([isinstance(d, Data) for d in data.values()])
    assert all(
        [
            d.x.shape == torch.Size((20, 10))
            for d in data.values()
        ]
    )
    assert all(
        [
            d.y.shape == torch.Size((20, 10))
            for d in data.values()
        ]
    )
    assert all(
        [
            d.edge_index.shape == torch.Size((2, 60))
            for d in data.values()
        ]
    )
    assert all(
        [d.edge_attr.shape[0] == 60 for d in data.values()]
    )

    data = dataset.fetch_from_idx_list([i for i in range(20)])
    assert isinstance(data, dict)
    assert all([isinstance(d["input_points"], Data) for d in data.values()])
    assert all(
        [isinstance(d["output_points"], torch.Tensor) for d in data.values()]
    )
    assert all(
        [
            d["input_points"].x.shape == torch.Size((400, 10))
            for d in data.values()
        ]
    )
    assert all(
        [
            d["output_points"].shape == torch.Size((400, 10))
            for d in data.values()
        ]
    )
    assert all(
        [
            d["input_points"].edge_index.shape == torch.Size((2, 1200))
            for d in data.values()
        ]
    )
    assert all(
        [d["input_points"].edge_attr.shape[0] == 1200 for d in data.values()]
    )
