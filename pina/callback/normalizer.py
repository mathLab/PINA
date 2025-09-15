"""Module for the Normalizer callback."""

import torch
from lightning.pytorch import Callback
from ..label_tensor import LabelTensor
from ..utils import check_consistency
from ..condition import InputTargetCondition

_REQUIRED_KEYS = {"scale", "shift"}


class NormalizerDataCallback(Callback):
    r"""
    A Lightning Callback that normalizes dataset inputs or targets
    according to user-provided scale and shift parameters.

    The transformation is applied as:

    .. math::

        x_{\text{new}} = \frac{x - \text{shift}}{\text{scale}}

    :Example:

    >>> NormalizerDataCallback()
    >>> NormalizerDataCallback(
    ...     "scale": torch.var,
    ...     "shift": torch.median
    ... )

    """

    def __init__(
        self,
        scale_fn=torch.std,
        shift_fn=torch.mean,
        stage="all",
        apply_to="input",
    ):
        """
        Initialize the NormalizerDataCallback.

        :param dict strategy: Normalization specification. It must be a dict
            with keys "scale" and "shift", each mapping to a callable that
            computes the respective value from a tensor. If None, defaults to
            using mean and std. Defaults is ``None``.
        :param str stage: Stage during which to apply normalization.
            One of {"train", "validate", "test", "all"}.
            Defaults to "all".
        :param str apply_to: Whether to normalize "input" or "target" data.
            Defaults to "input".
        :raises ValueError: If `apply_to` or `stage` are invalid.
        """
        super().__init__()

        self.apply_to = self._validate_apply_to(apply_to)
        self.stage = self._validate_stage(stage)
        if not callable(scale_fn):
            raise ValueError(f"scale_fn must be callable, got {scale_fn}")
        self.scale_fn = scale_fn
        if not callable(shift_fn):
            raise ValueError(f"shift_fn must be callable, got {shift_fn}")
        self.shift_fn = shift_fn
        self.normalizer = {}

    def _validate_apply_to(self, apply_to):
        """
        Validate the `apply_to` parameter.

        :param str apply_to: Candidate value for `apply_to`.
        :raises ValueError: If `apply_to` is not "input" or "target".
        :return: Validated `apply_to` value.
        :rtype: str
        """
        check_consistency(apply_to, str)
        if apply_to not in {"input", "target"}:
            raise ValueError(
                f"apply_to must be 'input' or 'target', got {apply_to}"
            )
        return apply_to

    def _validate_stage(self, stage):
        """
        Validate the `stage` parameter.

        :param str stage: Candidate value for `stage`.
        :raises ValueError: If `stage` is not one of "train", "validate",
            "test", or "all".
        :return: Validated `stage` value.
        :rtype: str
        """
        check_consistency(stage, str)
        if stage not in {"train", "validate", "test", "all"}:
            raise ValueError(
                f"stage must be 'train', 'validate', 'test', or 'all', got "
                f"{stage}"
            )
        return stage

    def setup(self, trainer, pl_module, stage):
        """
        Apply normalization during setup.

        :param Trainer trainer: A :class:`~pina.trainer.Trainer` instance.
        :param SolverInterface pl_module: A
            :class:`~pina.solver.solver.SolverInterface` instance.
        :param str stage: Current stage, not used kept for consistency.
        :raises RuntimeError: If condition names do not match solver conditions.
        :raises RuntimeError: If attempting to scale unavailable targets.
        :return: Result of parent setup.
        :rtype: Any
        """
        # extract conditions
        conditions_to_normalize = []
        for name, cond in pl_module.problem.conditions.items():
            if isinstance(cond, InputTargetCondition):
                conditions_to_normalize.append(name)

        if not self.normalizer:
            if not trainer.datamodule.train_dataset:
                raise RuntimeError(
                    "Training dataset is not available. Cannot compute "
                    "normalization parameters."
                )
            self._compute_scale_shift(
                conditions_to_normalize, trainer.datamodule.train_dataset
            )

        if stage == "fit" and self.stage in ["train", "all"]:
            self._scale_data(trainer.datamodule.train_dataset)
        if stage == "fit" and self.stage in ["validate", "all"]:
            self._scale_data(trainer.datamodule.val_dataset)
        if stage == "test" and self.stage in ["test", "all"]:
            self._scale_data(trainer.datamodule.test_dataset)
        return super().setup(trainer, pl_module, stage)

    def _compute_scale_shift(self, conditions, dataset):
        """
        Compute scale and shift for each condition from dataset.

        :param list conditions: List of condition names.
        :param dataset: `~pina.data.dataset.PinaDataset` object.
        :rtype: dict
        """
        for cond in conditions:
            if cond in dataset.conditions_dict:
                data = dataset.conditions_dict[cond][self.apply_to]
                shift = self.shift_fn(data)
                scale = self.scale_fn(data)
                self.normalizer[cond] = {
                    "shift": shift,
                    "scale": scale,
                }

    @staticmethod
    def _norm_fn(value, scale, shift):
        """
        Normalize a tensor with the given scale and shift.

        :param value: Input tensor to normalize.
        :type value: torch.Tensor | LabelTensor
        :param scale: Scaling factor.
        :type scale: float | int
        :param shift: Shifting factor.
        :type shift: float | int
        :return: Normalized tensor (value - shift) / scale.
        :rtype: torch.Tensor | LabelTensor
        """
        scaled_value = (value - shift) / scale
        if isinstance(value, LabelTensor):
            scaled_value = LabelTensor(scaled_value, value.labels)
        return scaled_value

    def _scale_data(self, dataset):
        """
        Apply normalization to a dataset in-place.

        :param dataset: Dataset object with `conditions_dict` and `update_data`.
        :type dataset: object
        """
        new_points = {}
        for cond, norm_params in self.normalizer.items():
            current_points = dataset.conditions_dict[cond][self.apply_to]
            scale = norm_params["scale"]
            shift = norm_params["shift"]
            new_points[cond] = {
                self.apply_to: self._norm_fn(current_points, scale, shift)
            }
        dataset.update_data(new_points)
