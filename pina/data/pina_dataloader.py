import torch

from .sample_dataset import SamplePointDataset
from .data_dataset import DataPointDataset
from .pina_batch import Batch


class SamplePointLoader:
    """
    This class is used to create a dataloader to use during the training.

    :var condition_names: The names of the conditions. The order is consistent
        with the condition indeces in the batches.
    :vartype condition_names: list[str]
    """

    def __init__(
        self, sample_dataset, data_dataset, batch_size=None, shuffle=True
    ) -> None:
        """
        Constructor.

        :param SamplePointDataset sample_pts: The sample points dataset.
        :param int batch_size: The batch size. If ``None``, the batch size is
            set to the number of sample points. Default is ``None``.
        :param bool shuffle: If ``True``, the sample points are shuffled.
            Default is ``True``.
        """
        if not isinstance(sample_dataset, SamplePointDataset):
            raise TypeError(
                f"Expected SamplePointDataset, got {type(sample_dataset)}"
            )
        if not isinstance(data_dataset, DataPointDataset):
            raise TypeError(
                f"Expected DataPointDataset, got {type(data_dataset)}"
            )

        self.n_data_conditions = len(data_dataset.condition_names)
        self.n_phys_conditions = len(sample_dataset.condition_names)
        data_dataset.condition_indeces += self.n_phys_conditions

        self._prepare_sample_dataset(sample_dataset, batch_size, shuffle)
        self._prepare_data_dataset(data_dataset, batch_size, shuffle)

        self.condition_names = (
            sample_dataset.condition_names + data_dataset.condition_names
        )

        self.batch_list = []
        for i in range(len(self.batch_sample_pts)):
            self.batch_list.append(("sample", i))

        for i in range(len(self.batch_input_pts)):
            self.batch_list.append(("data", i))

        if shuffle:
            self.random_idx = torch.randperm(len(self.batch_list))
        else:
            self.random_idx = torch.arange(len(self.batch_list))

        self._prepare_batches()

    def _prepare_data_dataset(self, dataset, batch_size, shuffle):
        """
        Prepare the dataset for data points.

        :param SamplePointDataset dataset: The dataset.
        :param int batch_size: The batch size.
        :param bool shuffle: If ``True``, the sample points are shuffled.
        """
        self.sample_dataset = dataset

        if len(dataset) == 0:
            self.batch_data_conditions = []
            self.batch_input_pts = []
            self.batch_output_pts = []
            return

        if batch_size is None:
            batch_size = len(dataset)
        batch_num = len(dataset) // batch_size
        if len(dataset) % batch_size != 0:
            batch_num += 1

        output_labels = dataset.output_pts.labels
        input_labels = dataset.input_pts.labels
        self.tensor_conditions = dataset.condition_indeces

        if shuffle:
            idx = torch.randperm(dataset.input_pts.shape[0])
            self.input_pts = dataset.input_pts[idx]
            self.output_pts = dataset.output_pts[idx]
            self.tensor_conditions = dataset.condition_indeces[idx]

        self.batch_input_pts = torch.tensor_split(dataset.input_pts, batch_num)
        self.batch_output_pts = torch.tensor_split(
            dataset.output_pts, batch_num
        )
        #print(input_labels)
        for i in range(len(self.batch_input_pts)):
            self.batch_input_pts[i].labels = input_labels
            self.batch_output_pts[i].labels = output_labels

        self.batch_data_conditions = torch.tensor_split(
            self.tensor_conditions, batch_num
        )

    def _prepare_sample_dataset(self, dataset, batch_size, shuffle):
        """
        Prepare the dataset for sample points.

        :param DataPointDataset dataset: The dataset.
        :param int batch_size: The batch size.
        :param bool shuffle: If ``True``, the sample points are shuffled.
        """

        self.sample_dataset = dataset
        if len(dataset) == 0:
            self.batch_sample_conditions = []
            self.batch_sample_pts = []
            return

        if batch_size is None:
            batch_size = len(dataset)

        batch_num = len(dataset) // batch_size
        if len(dataset) % batch_size != 0:
            batch_num += 1

        self.tensor_pts = dataset.pts
        self.tensor_conditions = dataset.condition_indeces

        # if shuffle:
        #     idx = torch.randperm(self.tensor_pts.shape[0])
        #     self.tensor_pts = self.tensor_pts[idx]
        #     self.tensor_conditions = self.tensor_conditions[idx]

        self.batch_sample_pts = torch.tensor_split(self.tensor_pts, batch_num)
        for i in range(len(self.batch_sample_pts)):
            self.batch_sample_pts[i].labels = dataset.pts.labels

        self.batch_sample_conditions = torch.tensor_split(
            self.tensor_conditions, batch_num
        )

    def _prepare_batches(self):
        """
        Prepare the batches.
        """
        self.batches = []
        for i in range(len(self.batch_list)):
            type_, idx_ = self.batch_list[i]

            if type_ == "sample":
                batch = Batch(
                    "sample", idx_,
                    self.batch_sample_pts,
                    self.batch_sample_conditions)
            else:
                batch = Batch(
                    "data", idx_,
                    self.batch_input_pts,
                    self.batch_output_pts,
                    self.batch_data_conditions)
                
            self.batches.append(batch)

    def __iter__(self):
        """
        Return an iterator over the points. Any element of the iterator is a
        dictionary with the following keys:
            - ``pts``: The input sample points. It is a LabelTensor with the
                shape ``(batch_size, input_dimension)``.
            - ``output``: The output sample points. This key is present only
                if data conditions are present. It is a LabelTensor with the
                shape ``(batch_size, output_dimension)``.
            - ``condition``: The integer condition indeces. It is a tensor
                with the shape ``(batch_size, )`` of type ``torch.int64`` and
                indicates for any ``pts`` the corresponding problem condition.

        :return: An iterator over the points.
        :rtype: iter
        """
        # for i in self.random_idx:
        for i in self.random_idx:
            yield self.batches[i]

        # for i in range(len(self.batch_list)):
        #     type_, idx_ = self.batch_list[i]

        #     if type_ == "sample":
        #         d = {
        #             "pts": self.batch_sample_pts[idx_].requires_grad_(True),
        #             "condition": self.batch_sample_conditions[idx_],
        #         }
        #     else:
        #         d = {
        #             "pts": self.batch_input_pts[idx_].requires_grad_(True),
        #             "output": self.batch_output_pts[idx_],
        #             "condition": self.batch_data_conditions[idx_],
        #         }
        #     yield d

    def __len__(self):
        """
        Return the number of batches.

        :return: The number of batches.
        :rtype: int
        """
        return len(self.batch_list)
