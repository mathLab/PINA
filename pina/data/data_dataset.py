from torch.utils.data import Dataset
import torch
from ..label_tensor import LabelTensor


class DataPointDataset(Dataset):

    def __init__(self, problem, device) -> None:
        super().__init__()
        input_list = []
        output_list = []
        self.condition_names = []

        for name, condition in problem.conditions.items():
            if hasattr(condition, "output_points"):
                input_list.append(problem.conditions[name].input_points)
                output_list.append(problem.conditions[name].output_points)
                self.condition_names.append(name)

        self.input_pts = LabelTensor.cat(input_list)
        self.output_pts = LabelTensor.cat(output_list)

        if self.input_pts != []:
            self.condition_indeces = torch.cat(
                [
                    torch.tensor([i] * len(input_list[i]))
                    for i in range(len(self.condition_names))
                ],
                dim=0,
            )
        else:  # if there are no data points
            self.condition_indeces = torch.tensor([])
            self.input_pts = torch.tensor([])
            self.output_pts = torch.tensor([])

        self.input_pts = self.input_pts.to(device)
        self.output_pts = self.output_pts.to(device)
        self.condition_indeces = self.condition_indeces.to(device)

    def __len__(self):
        return self.input_pts.shape[0]