from torch.utils.data import Dataset, DataLoader
import functools


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



# TODO: working also for datapoints
class DummyLoader:

    def __init__(self, data, device) -> None:

        # TODO: We need to make a dataset somehow
        #       and the PINADataset needs to have a method
        #       to send points to device
        #       now we simply do it here
        # send data to device
        def convert_tensors(pts, device):
            pts = pts.to(device)
            pts.requires_grad_(True)
            pts.retain_grad()
            return pts

        for location, pts in data.items():
            if isinstance(pts, (tuple, list)):
                pts = tuple(map(functools.partial(convert_tensors, device=device),pts))
            else:
                pts = pts.to(device)
                pts = pts.requires_grad_(True)
                pts.retain_grad()
            
            data[location] = pts

        # iterator
        self.data = [data]

    def __iter__(self):
        return iter(self.data)
