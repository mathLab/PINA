"""Module for the abstract SolverInterface base class."""

from abc import ABCMeta, abstractmethod
import lightning


class SolverInterface(lightning.pytorch.LightningModule, metaclass=ABCMeta):
    """
    Abstract base class for PINA solvers. All specific solvers must inherit
    from this interface. This class extends
    :class:`~lightning.pytorch.core.LightningModule`, providing additional
    functionalities for defining and optimizing Deep Learning models.

    By inheriting from this base class, solvers gain access to built-in training
    loops, logging utilities, and optimization techniques.
    """

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Abstract method for the forward pass implementation.

        :param args: The input tensor.
        :type args: torch.Tensor | LabelTensor | Data | Graph
        :param dict kwargs: Additional keyword arguments.
        """

    @abstractmethod
    def optimization_cycle(self, batch):
        """
        The optimization cycle for the solvers.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :return: The losses computed for all conditions in the batch, casted
            to a subclass of :class:`torch.Tensor`. It should return a dict
            containing the condition name and the associated scalar loss.
        :rtype: dict
        """

    @abstractmethod
    def training_step(self, batch, **kwargs):
        """
        Solver training step. It computes the optimization cycle and aggregates
        the losses using the ``weighting`` attribute.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param dict kwargs: Additional keyword arguments passed to
            ``optimization_cycle``.
        :return: The loss of the training step.
        :rtype: torch.Tensor
        """

    @abstractmethod
    def validation_step(self, batch, **kwargs):
        """
        Solver validation step. It computes the optimization cycle and
        averages the losses. No aggregation using the ``weighting`` attribute is
        performed.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param dict kwargs: Additional keyword arguments passed to
            ``optimization_cycle``.
        :return: The loss of the training step.
        :rtype: torch.Tensor
        """

    @abstractmethod
    def test_step(self, batch, **kwargs):
        """
        Solver test step. It computes the optimization cycle and
        averages the losses. No aggregation using the ``weighting`` attribute is
        performed.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param dict kwargs: Additional keyword arguments passed to
            ``optimization_cycle``.
        :return: The loss of the training step.
        :rtype: torch.Tensor
        """

    @abstractmethod
    def setup(self, stage):
        """
        This method is called at the start of the train and test process to
        compile the model if the :class:`~pina.trainer.Trainer`
        ``compile`` is ``True``.

        :param str stage: The current stage of the training process
            (e.g., ``fit``, ``validate``, ``test``, ``predict``).
        :return: The result of the parent class ``setup`` method.
        :rtype: Any
        """
