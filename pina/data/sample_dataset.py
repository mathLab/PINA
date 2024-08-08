from torch.utils.data import Dataset
import torch

from ..label_tensor import LabelTensor


class SamplePointDataset(Dataset):
    """
    This class is used to create a dataset of sample points.
    """

    def __init__(self, problem, device) -> None:
        """
        :param dict input_pts: The input points.
        """
        super().__init__()
        pts_list = []
        self.condition_names = []

        for name, condition in problem.conditions.items():
            if not hasattr(condition, "output_points"):
                pts_list.append(problem.input_pts[name])
                self.condition_names.append(name)

        self.pts = LabelTensor.stack(pts_list)

        if self.pts != []:
            self.condition_indeces = torch.cat(
                [
                    torch.tensor([i] * len(pts_list[i]))
                    for i in range(len(self.condition_names))
                ],
                dim=0,
            )
        else:  # if there are no sample points
            self.condition_indeces = torch.tensor([])
            self.pts = torch.tensor([])

        self.pts = self.pts.to(device)
        self.condition_indeces = self.condition_indeces.to(device)

    def __len__(self):
        return self.pts.shape[0]