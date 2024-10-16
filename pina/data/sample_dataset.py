from torch.utils.data import Dataset
import torch
from ..label_tensor import LabelTensor


class SamplePointDataset(Dataset):
    """
    This class is used to create a dataset of sample points.
    """

    def __init__(self, problem, device) -> None:
        """
        TODO
        """
        pts_list = []
        self.condition_names = {}
        collector = problem.collector
        idx = 0
        for name, data in collector.data_collections.items():
            if 'output_points' not in data.keys() and 'input_points' in data.keys():
                pts_list.append(data['input_points'])
                self.condition_names[idx] = name
                idx += 1

        self.pts = LabelTensor.vstack(pts_list) if len(pts_list) > 0 else None
        if self.pts is not None:
            self.condition_indices = torch.cat(
                [
                    torch.tensor([i] * len(pts_list[i]), dtype=torch.uint8)
                    for i in range(len(self.condition_names))
                ],
                dim=0,
            )
        else:  # if there are no sample points
            self.condition_indices = torch.tensor([])
            self.pts = torch.tensor([])
        self.pts = self.pts.to(device)
        self.condition_indices = self.condition_indices.to(device)
        self.splitting_dimension = 0

    def __len__(self):
        """
        TODO
        """
        return self.pts.shape[0]

    def __getitem__(self, idx):
        """
        TODO
        """
        return self.pts[[idx]], self.condition_indices[idx]

