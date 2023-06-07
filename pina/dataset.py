class PinaDataset():

    def __init__(self, pinn) -> None:
        self.pinn = pinn

    @property
    def dataloader(self):
        return self._create_dataloader()

    @property
    def dataset(self):
        return [self.SampleDataset(key, val)
                for key, val in self.input_pts.items()]

    def _create_dataloader(self):
        """Private method for creating dataloader

        :return: dataloader
        :rtype: torch.utils.data.DataLoader
        """
        if self.pinn.batch_size is None:
            return {key: [{key: val}] for key, val in self.pinn.input_pts.items()}

        def custom_collate(batch):
            # extracting pts labels
            _, pts = list(batch[0].items())[0]
            labels = pts.labels
            # calling default torch collate
            collate_res = default_collate(batch)
            # save collate result in dict
            res = {}
            for key, val in collate_res.items():
                val.labels = labels
                res[key] = val
        def __getitem__(self, index):
            tensor = self._tensor.select(0, index)
            return {self._location: tensor}

        def __len__(self):
            return self._len

from torch.utils.data import Dataset, DataLoader
class LabelTensorDataset(Dataset):
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)
        self.labels = list(d.keys())
        
    def __getitem__(self, index):
        print(index)
        result = {}
        for label in self.labels:
            sample_tensor = getattr(self, label)

            # print('porcodio')
            # print(sample_tensor.shape[1])
            # print(index)
            # print(sample_tensor[index])
            try:
                result[label] = sample_tensor[index]
            except IndexError:
                result[label] = torch.tensor([])
        
        print(result)
        return result

    def __len__(self):
        return max([len(getattr(self, label)) for label in self.labels])


class LabelTensorDataLoader(DataLoader):

    def collate_fn(self, data):
        print(data)
        gggggggggg
#         return dict(zip(self.pinn.input_pts.keys(), dataloaders))

#     class SampleDataset(torch.utils.data.Dataset):

#         def __init__(self, location, tensor):
#             self._tensor = tensor
#             self._location = location
#             self._len = len(tensor)

#         def __getitem__(self, index):
#             tensor = self._tensor.select(0, index)
#             return {self._location: tensor}

#         def __len__(self):
#             return self._len

from torch.utils.data import Dataset, DataLoader
class LabelTensorDataset(Dataset):
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)
        self.labels = list(d.keys())
        
    def __getitem__(self, index):
        print(index)
        result = {}
        for label in self.labels:
            sample_tensor = getattr(self, label)

            # print('porcodio')
            # print(sample_tensor.shape[1])
            # print(index)
            # print(sample_tensor[index])
            try:
                result[label] = sample_tensor[index]
            except IndexError:
                result[label] = torch.tensor([])
        
        print(result)
        return result

    def __len__(self):
        return max([len(getattr(self, label)) for label in self.labels])

# TODO: working also for datapoints
class DummyLoader:

    def __init__(self, data) -> None:
        self.data = [data]

    def __iter__(self):
        return iter(self.data)