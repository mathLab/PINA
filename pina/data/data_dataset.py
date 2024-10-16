from torch.utils.data import Dataset
import torch
from ..label_tensor import LabelTensor

class DataPointDataset(Dataset):
    def __init__(self, problem, device) -> None:
        super().__init__()
        input_list = []
        output_list = []
        self.condition_names = {}
        collector = problem.collector
        idx = 0
        for name, data in collector.data_collections.items():
            if 'output_points' in data.keys():
                input_list.append(data['input_points'])
                output_list.append(data['output_points'])
                self.condition_names[idx] = name
                idx += 1

        self.input_pts = LabelTensor.vstack(input_list) if len(input_list) > 0 else None
        self.output_pts = LabelTensor.vstack(output_list) if len(input_list) > 0 else None

        if self.input_pts is not None:
            self.condition_indices = torch.cat(
                [
                    torch.tensor([i] * len(input_list[i]), dtype=torch.uint8)
                    for i in range(len(self.condition_names))
                ],
                dim=0,
            )
        else:  # if there are no data points
            self.condition_indices = torch.tensor([])
            self.input_pts = torch.tensor([])
            self.output_pts = torch.tensor([])

        self.input_pts = self.input_pts.to(device)
        self.output_pts = self.output_pts.to(device)
        self.condition_indices = self.condition_indices.to(device)

    def __len__(self):
        return self.input_pts.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return getattr(self, idx)
        elif isinstance(idx, (tuple, list)):
            tensor = getattr(self, idx[0])
            return tensor[[idx[1]]]