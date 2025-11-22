# import torch
# import pytest
# from pina.data.dataset import PinaDatasetFactory, PinaTensorDataset

# input_tensor = torch.rand((100, 10))
# output_tensor = torch.rand((100, 2))

# input_tensor_2 = torch.rand((50, 10))
# output_tensor_2 = torch.rand((50, 2))

# conditions_dict_single = {
#     "data": {
#         "input": input_tensor,
#         "target": output_tensor,
#     }
# }

# conditions_dict_single_multi = {
#     "data_1": {
#         "input": input_tensor,
#         "target": output_tensor,
#     },
#     "data_2": {
#         "input": input_tensor_2,
#         "target": output_tensor_2,
#     },
# }

# max_conditions_lengths_single = {"data": 100}

# max_conditions_lengths_multi = {"data_1": 100, "data_2": 50}


# @pytest.mark.parametrize(
#     "conditions_dict, max_conditions_lengths",
#     [
#         (conditions_dict_single, max_conditions_lengths_single),
#         (conditions_dict_single_multi, max_conditions_lengths_multi),
#     ],
# )
# def test_constructor_tensor(conditions_dict, max_conditions_lengths):
#     dataset = PinaDatasetFactory(
#         conditions_dict,
#         max_conditions_lengths=max_conditions_lengths,
#         automatic_batching=True,
#     )
#     assert isinstance(dataset, PinaTensorDataset)


# def test_getitem_single():
#     dataset = PinaDatasetFactory(
#         conditions_dict_single,
#         max_conditions_lengths=max_conditions_lengths_single,
#         automatic_batching=False,
#     )

#     tensors = dataset.fetch_from_idx_list([i for i in range(70)])
#     assert isinstance(tensors, dict)
#     assert list(tensors.keys()) == ["data"]
#     assert sorted(list(tensors["data"].keys())) == ["input", "target"]
#     assert isinstance(tensors["data"]["input"], torch.Tensor)
#     assert tensors["data"]["input"].shape == torch.Size((70, 10))
#     assert isinstance(tensors["data"]["target"], torch.Tensor)
#     assert tensors["data"]["target"].shape == torch.Size((70, 2))


# def test_getitem_multi():
#     dataset = PinaDatasetFactory(
#         conditions_dict_single_multi,
#         max_conditions_lengths=max_conditions_lengths_multi,
#         automatic_batching=False,
#     )
#     tensors = dataset.fetch_from_idx_list([i for i in range(70)])
#     assert isinstance(tensors, dict)
#     assert list(tensors.keys()) == ["data_1", "data_2"]
#     assert sorted(list(tensors["data_1"].keys())) == ["input", "target"]
#     assert isinstance(tensors["data_1"]["input"], torch.Tensor)
#     assert tensors["data_1"]["input"].shape == torch.Size((70, 10))
#     assert isinstance(tensors["data_1"]["target"], torch.Tensor)
#     assert tensors["data_1"]["target"].shape == torch.Size((70, 2))

#     assert sorted(list(tensors["data_2"].keys())) == ["input", "target"]
#     assert isinstance(tensors["data_2"]["input"], torch.Tensor)
#     assert tensors["data_2"]["input"].shape == torch.Size((50, 10))
#     assert isinstance(tensors["data_2"]["target"], torch.Tensor)
#     assert tensors["data_2"]["target"].shape == torch.Size((50, 2))
