"""Module for the DeepEnsemble simple solver."""

from pina._src.solver.multi_model_simple_solver import MultiModelSimpleSolver
from pina._src.core.utils import check_consistency


class EnsembleSimpleSolver(MultiModelSimpleSolver):
    r"""
    Ensemble Simple Solver class. This class implements an ensemble
    solver for generic conditions (data, equations, or domain residuals) using
    user-specified ``models`` to solve a specific ``problem``.

    It is the ensemble counterpart of
    :class:`~pina.solver.SingleModelSimpleSolver`: each model in the ensemble
    evaluates every condition independently, and the per-model scalar losses
    are averaged to produce the final condition loss.

    An ensemble model is constructed by combining multiple models that solve
    the same type of problem. Mathematically, this creates an implicit
    distribution :math:`p(\mathbf{u} \mid \mathbf{s})` over the possible
    outputs :math:`\mathbf{u}`, given the original input :math:`\mathbf{s}`.
    The models :math:`\mathcal{M}_{i\in (1,\dots,r)}` in
    the ensemble work collaboratively to capture different
    aspects of the data or task, with each model contributing a distinct
    prediction
    :math:`\mathbf{y}_{i}=\mathcal{M}_i(\mathbf{u} \mid \mathbf{s})`.
    By aggregating these predictions, the ensemble
    model can achieve greater robustness and accuracy compared to individual
    models, leveraging the diversity of the models to reduce overfitting and
    improve generalization. Furthemore, statistical metrics can
    be computed, e.g. the ensemble mean and variance:

    .. math::
        \mathbf{\mu} = \frac{1}{N}\sum_{i=1}^r \mathbf{y}_{i}

    .. math::
        \mathbf{\sigma^2} = \frac{1}{N}\sum_{i=1}^r
        (\mathbf{y}_{i} - \mathbf{\mu})^2

    During training the condition loss is minimised by each ensemble model
    independently and then averaged:

    .. math::
        \mathcal{L}_{\rm{condition}} = \frac{1}{N_{\rm{ensemble}}}
        \sum_{i=1}^{N_{\rm{ensemble}}}
        \mathcal{L}_i(\mathcal{M}_i, \mathbf{s})

    where :math:`\mathcal{L}` is a specific loss function, typically the MSE:

    .. math::
        \mathcal{L}(v) = \| v \|^2_2.

    .. seealso::

        **Original reference**: Lakshminarayanan, B., Pritzel, A., & Blundell,
        C. (2017). *Simple and scalable predictive uncertainty estimation
        using deep ensembles*. Advances in neural information
        processing systems, 30.
        DOI: `arXiv:1612.01474 <https://arxiv.org/abs/1612.01474>`_.
    """

    def __init__(
        self,
        problem,
        models,
        optimizers=None,
        schedulers=None,
        weighting=None,
        loss=None,
        use_lt=True,
    ):
        """
        Initialization of the :class:`DeepEnsembleSimpleSolver` class.

        :param BaseProblem problem: The problem to be solved.
        :param list[torch.nn.Module] models: The neural network models to be
            used. Must be a list or tuple with at least two models.
        :param list[OptimizerInterface] optimizers: The optimizers to be used.
            If ``None``, the :class:`torch.optim.Adam` optimizer is used for
            each model. Default is ``None``.
        :param list[SchedulerInterface] schedulers: The learning rate
            schedulers. If ``None`` :class:`torch.optim.lr_scheduler.ConstantLR`
            is used for each model. Default is ``None``.
        :param WeightingInterface weighting: The weighting schema to be used.
            If ``None``, no weighting schema is used. Default is ``None``.
        :param torch.nn.Module loss: The element-wise loss module.
            If ``None``, :class:`torch.nn.MSELoss` is used. Default is
            ``None``.
        :param bool use_lt: If ``True``, the solver uses LabelTensors as
            input. Default is ``True``.
        :param int ensemble_dim: The dimension along which the per-model
            outputs are stacked in :meth:`forward`. Default is ``0``.
        """
        MultiModelSimpleSolver.__init__(
            self,
            problem=problem,
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            weighting=weighting,
            loss=loss,
            use_lt=use_lt,
        )

    @property
    def num_ensemble(self):
        """
        The number of models in the ensemble.

        :return: The number of models in the ensemble.
        :rtype: int
        """
        return len(self.models)