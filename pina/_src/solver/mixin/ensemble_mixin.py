import torch
from pina._src.solver.mixin.multi_model_mixin import _MultiModelMixin


class _EnsembleMixin(_MultiModelMixin):
    """
    Mixin that defines the forward pass and optimizer configuration for solvers
    backed by an ensemble of models. Provides properties to access the models,
    optimizers, and schedulers.

    Designed to be used in combination with any solver inheriting from
    :class:`~pina._src.solver.base_solver.BaseSolver`.
    """

    def forward(self, x):
        """
        The forward pass implementation that evaluates all models and returns
        the average of their outputs.

        :param x: The input data.
        :type x: torch.Tensor | LabelTensor | Data | Graph
        :return: The output of all models stacked together.
        :rtype: torch.Tensor | LabelTensor | Data | Graph
        """
        return torch.stack(
            [self.models[idx](x) for idx in range(self.num_models)]
        ).mean(dim=0)
