"""Module for the EquationConditionBase class."""

from pina._src.condition.condition_base import ConditionBase


class EquationConditionBase(ConditionBase):
    """
    Base class for conditions that involve an equation.

    This class provides the :meth:`evaluate` method, which computes the
    non-aggregated residual of the equation given the input samples and a
    solver. It is intended to be subclassed by conditions that define an
    ``equation`` attribute, such as
    :class:`~pina.condition.DomainEquationCondition` and
    :class:`~pina.condition.InputEquationCondition`.
    """

    def evaluate(self, batch, solver, loss):
        """
        Evaluate the equation residual on the given batch using the solver.

        This method computes the non-aggregated, element-wise residual of the
        equation. It performs a forward pass of the solver's model on the
        input samples and then evaluates the equation residual. The returned
        tensor is **not** reduced (i.e., no mean, sum, etc.), preserving the
        per-sample residual values.

        :param batch: The batch containing the ``input`` entry.
        :type batch: dict | _DataManager
        :param solver: The solver containing the model and any additional
            parameters (e.g., unknown parameters for inverse problems).
        :type solver: ~pina.solver.solver.SolverInterface
        :param loss: The non-aggregating loss function to apply to the
            computed residual against zero.
        :type loss: torch.nn.Module
        :return: The non-aggregated loss tensor.
        :rtype: ~pina.label_tensor.LabelTensor

        :Example:

            >>> residuals = condition.evaluate(
            ...     {"input": input_samples}, solver, loss
            ... )
            >>> # residuals is a non-reduced tensor of shape (n_samples, ...)
        """
        samples = batch["input"].requires_grad_(True)
        print("samples", samples)
        residual = self.equation.residual(
            samples, solver.forward(samples), solver._params
        )
        # assert False
        return residual**2
