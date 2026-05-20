"""Module for the Condition interface."""

from abc import ABCMeta, abstractmethod


class ConditionInterface(metaclass=ABCMeta):
    """
    Abstract interface for all conditions.

    Refer to :class:`pina.condition.condition.Condition` for a thorough
    description of all available conditions and how to instantiate them.
    """

    @abstractmethod
    def __len__(self):
        """
        Return the number of data points in the condition.

        :return: The number of data points.
        :rtype: int
        """

    @abstractmethod
    def __getitem__(self, idx):
        """
        Return the data point at the specified index.

        :param int idx: The index of the data point to retrieve.
        :return: The data point at the specified index.
        :rtype: Any
        """

    @abstractmethod
    def store_data(self, **kwargs):
        """
        Store the data for the condition in a suitable format.

        :param dict kwargs: The keyword arguments containing the data to be
            stored.
        :return: The stored data in a suitable format.
        :rtype: Any
        """

    @abstractmethod
    def create_dataloader(
        self, dataset, batch_size, automatic_batching, **kwargs
    ):
        """
        Create the DataLoader for the condition.

        :param Dataset dataset: The dataset for the DataLoader.
        :param int batch_size: The batch size for the DataLoader.
        :param bool automatic_batching: Whether to use automatic batching.
        :param dict kwargs: Additional keyword arguments for the DataLoader.
        :return: The DataLoader for the condition.
        :rtype: torch.utils.data.DataLoader
        """

    @abstractmethod
    def evaluate(self, batch, solver):
        """
        Evaluate the residual of the condition on the given batch using the
        solver.

        This method computes the non-aggregated, element-wise residual of the
        condition. A forward pass of the solver's model is performed on the
        input samples, and the condition residual is evaluated accordingly.

        The returned tensor is not reduced, preserving the per-sample residual
        values.

        :param dict batch: The batch containing the data required by the
            condition evaluation.
        :param SolverInterface solver: The solver used to perform the forward
            pass and compute the residual. The solver provides access to the
            model and its parameters, which may be necessary for evaluating the
            condition residual.
        :return: The non-aggregated residual tensor.
        :rtype: torch.Tensor | LabelTensor
        """

    @abstractmethod
    def switch_dataloader_fn(self, create_dataloader_fn):
        """
        Switch the dataloader function for the condition.

        :param Callable create_dataloader_fn: The new dataloader function to use
            for the condition.
        :return: The new dataloader function for the condition.
        :rtype: Callable
        """

    @classmethod
    @abstractmethod
    def automatic_batching_collate_fn(cls, batch):
        """
        Collate function for automatic batching to be used in the DataLoader.

        :param list batch: A list of items from the dataset.
        :return: A collated batch.
        :rtype: dict
        """

    @staticmethod
    @abstractmethod
    def collate_fn(batch, condition):
        """
        Collate function for custom batching to be used in the DataLoader.

        :param list batch: A list of items from the dataset.
        :param BaseCondition condition: The condition instance.
        :return: A collated batch.
        :rtype: dict
        """

    @property
    @abstractmethod
    def problem(self):
        """
        The problem associated with this condition.

        :return: The problem associated with this condition.
        :rtype: BaseProblem
        """

    @problem.setter
    @abstractmethod
    def problem(self, value):
        """
        Set the problem associated with this condition.

        :param BaseProblem value: The problem to associate with this condition.
        """
