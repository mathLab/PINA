import torch


class _PhysicsInformedMixin:
    """
    Mixin that enables physics-informed training by ensuring gradients are
    enabled during validation and testing, which is necessary for computing
    physics residuals.

    Designed to be used in combination with any solver inheriting from
    :class:`~pina._src.solver.base_solver.BaseSolver`.
    """

    @torch.enable_grad()
    def validation_step(self, batch, batch_idx):
        """
        Solver validation step.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param int batch_idx: The index of the current batch.
        :return: The loss of the training step.
        :rtype: torch.Tensor
        """
        return super().validation_step(batch, batch_idx)

    @torch.enable_grad()
    def test_step(self, batch, batch_idx):
        """
        Solver test step.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param int batch_idx: The index of the current batch.
        :return: The loss of the training step.
        :rtype: torch.Tensor
        """
        return super().test_step(batch, batch_idx)
