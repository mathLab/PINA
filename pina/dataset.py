from torch.utils.data import Dataset
import torch
from pina import LabelTensor


class SamplePointDataset(Dataset):
    """
    This class is used to create a dataset of sample points.
    """

    def __init__(self, input_pts) -> None:
        """
        :param dict input_pts: The input points.
        """
        super().__init__()
        self.pts = LabelTensor.vstack(list(input_pts.values()))

        self.conditions = torch.cat([
            torch.tensor([i]*len(pts))
            for i, pts in enumerate(input_pts.values())
        ], dim=0)
        self.label_encode = list(input_pts.keys())
        
        # self.pts.requires_grad_(True)
        # self.pts.retain_grad()
        # # print(self.pts)

    def __len__(self):
        return self.pts.shape[0]
    

class SamplePointLoader:
    """
    This class is used to create a dataloader to use during the training.
    """

    def __init__(self, sample_pts, batch_size=None, shuffle=True) -> None:
        """
        Constructor.

        :param SamplePointDataset sample_pts: The sample points dataset.
        :param int batch_size: The batch size. If ``None``, the batch size is
            set to the number of sample points. Default is ``None``.
        :param bool shuffle: If ``True``, the sample points are shuffled.
            Default is ``True``.
        """
        if not isinstance(sample_pts, SamplePointDataset):
            raise TypeError(f'Expected SamplePointDataset, got {type(sample_pts)}')

        if batch_size is None:
            batch_size = len(sample_pts)

        self.batch_size = batch_size
        self.batch_num = len(sample_pts) // batch_size
        
        self.tensor_pts = sample_pts.pts
        self.tensor_conditions = sample_pts.conditions

        if shuffle:
            idx = torch.randperm(self.tensor_pts.shape[0])
            self.tensor_pts = self.tensor_pts[idx]
            self.tensor_conditions = self.tensor_conditions[idx]
           
        self.tensor_pts = torch.tensor_split(self.tensor_pts, self.batch_num)
        for i, batch in enumerate(self.tensor_pts):
            self.tensor_pts[i].labels = sample_pts.pts.labels

        self.tensor_conditions = torch.tensor_split(
            self.tensor_conditions, self.batch_num)

    def __iter__(self):
        return iter(zip(self.tensor_pts, self.tensor_conditions))