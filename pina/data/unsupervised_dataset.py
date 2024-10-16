from torch.utils.data import Dataset
import torch
from ..label_tensor import LabelTensor


class UnsupervisedDataset(Dataset):
    """
    This class is used to create a dataset of unsupervised data points and, optionally, conditions.
    """

    def __init__(self, problem, device) -> None:
        """
        TODO
        """
        super().__init__()
        data = []
        conditional_variables = []
        self.condition_names = {}
        collector = problem.collector
        idx = 0
        for name, val in collector.data_collections.items():
            if 'data' in val.keys():
                data.append(val['input_points'])
                if 'conditional_variable' in val.keys():
                    conditional_variables.append(val['conditional_variable'])
                self.condition_names[idx] = name
                idx += 1

        self.data = LabelTensor.vstack(data) if len(data) > 0 else None
        self.conditional_variables = LabelTensor.vstack(conditional_variables) if len(conditional_variables) > 0 else None
        if self.data is not None:
            self.condition_indices = torch.cat(
                [
                    torch.tensor([i] * len(data[i]), dtype=torch.uint8)
                    for i in range(len(self.condition_names))
                ],
                dim=0,
            )
        else:  # if there are no sample points
            self.conditional_variables = torch.tensor([])
            self.condition_indices = torch.tensor([])
            self.data = torch.tensor([])
        self.data = self.data.to(device)
        self.condition_indices = self.condition_indices.to(device)

    def __len__(self):
        """
        TODO
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        TODO
        """
        return self.data[[idx]], self.conditional_variables[[idx]], self.condition_indices[idx]