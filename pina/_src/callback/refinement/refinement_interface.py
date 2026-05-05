"""Module for the Refinement Interface."""

from abc import ABCMeta, abstractmethod


class RefinementInterface(metaclass=ABCMeta):
    """
    Abstract interface for all refinement strategies.
    """

    @abstractmethod
    def on_train_start(self, trainer, solver):
        """
        This method is called once before training begins and is typically used
        to initialize datasets, sampling conditions, or internal state.

        :param Trainer trainer: The trainer managing the training loop.
        :param SolverInterface solver: The solver associated with the trainer.
        """

    @abstractmethod
    def on_train_epoch_end(self, trainer, solver):
        """
        Apply refinement at the end of a training epoch.

        This method is invoked after each epoch and can update the dataset based
        on the current state of the model.

        :param Trainer trainer: The trainer managing the training loop.
        :param SolverInterface solver: The solver associated with the trainer.
        """

    @abstractmethod
    def sample(self, current_points, condition_name, solver):
        """
        Generate new sample points for a given condition.

        :param LabelTensor current_points: The existing points in the domain.
        :param str condition_name: The identifier of the condition to refine.
        :param SolverInterface solver: The solver used for sampling decisions.
        :return: Newly sampled points.
        :rtype: LabelTensor
        """

    @property
    @abstractmethod
    def dataset(self):
        """
        The training dataset managed by the refinement strategy, which can be
        updated dynamically.

        :return: The current training dataset.
        :rtype: PinaDataset
        """

    @property
    @abstractmethod
    def initial_population_size(self):
        """
        Initial size of the sampled dataset for each condition before any
        refinement is applied.

        :return: A mapping between each condition name and its initial number
            of sampled points.
        :rtype: dict[str, int]
        """
