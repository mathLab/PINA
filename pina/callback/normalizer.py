"""Module for the Normalizer callback."""

import torch
from lightning.pytorch import Callback
from ..label_tensor import LabelTensor
from ..utils import check_consistency

_REQUIRED_KEYS = {"scale", "shift"}


class NormalizerDataCallback(Callback):
    r"""
    A Lightning Callback that normalizes dataset inputs or targets
    according to user-provided scale and shift parameters.

    The transformation is applied as:

    .. math::

        x_{\text{new}} = \frac{x - \text{shift}}{\text{scale}}

    :Example:

    >>> NormalizerDataCallback({"scale": 1, "shift": 0})
    >>> NormalizerDataCallback({
    ...     "a": {"scale": 2.0, "shift": 1.0},
    ...     "b": {"scale": 0.5, "shift": 0.0},
    ... })

    """

    def __init__(
        self,
        normalizer=None,
        stage="all",
        apply_to="input",
    ):
        """
        Initialize the NormalizerDataCallback.

        :param dict normalizer: Normalization specification. Either
            - a dict with
            {"scale": float | torch.Tensor, "shift": float | torch.Tensor}, or
            - a dict mapping condition names to such dicts. If ``None`` shift
            and scale parameters are inferred during runtime. Default ``None``.
        :param str stage: Stage during which to apply normalization.
            One of {"train", "validate", "test", "all"}.
            Defaults to "all".
        :param str apply_to: Whether to normalize "input" or "target" data.
            Defaults to "input".
        :raises ValueError: If `apply_to` or `stage` are invalid.
        """
        super().__init__()

        # validate apply_to
        check_consistency(apply_to, str)
        if apply_to not in {"input", "target"}:
            raise ValueError(
                f"apply_to must be 'input' or 'target', got {apply_to!r}"
            )

        # validate stage (can be None for setup flexibility)
        check_consistency(stage, str)
        if stage not in {"train", "validate", "test", "all"}:
            raise ValueError(
                f"stage must be 'train', 'validate', 'test', or 'all' "
                f"got {stage!r}"
            )

        self.normalizer = self._validate_normalizer(normalizer)
        self.apply_to = apply_to
        self.stage = stage

    def _is_normalizer_dict(self, d):
        """
        Check if a dictionary is a valid normalizer specification.

        :param dict d: Dictionary to validate.
        :return: True if dict has {"scale", "shift"} keys with numeric values.
        :rtype: bool
        """
        return (
            isinstance(d, dict)
            and set(d.keys()) == _REQUIRED_KEYS
            and all(
                isinstance(d[k], (float, int, torch.Tensor))
                for k in _REQUIRED_KEYS
            )
        )

    def _validate_normalizer(self, normalizer):
        """
        Validate a normalizer configuration.

        :param dict normalizer: Candidate normalizer specification.
        :raises ValueError: If the normalizer format is invalid.
        :return: A validated normalizer dictionary.
        :rtype: dict
        """
        if normalizer is None:
            return None

        if self._is_normalizer_dict(normalizer):
            return normalizer

        if isinstance(normalizer, dict) and all(
            self._is_normalizer_dict(v) for v in normalizer.values()
        ):
            return normalizer

        raise ValueError(
            "normalizer must be either:\n"
            f"  - dict with {_REQUIRED_KEYS} as keys\n"
            f"  - dict of dicts where the outer keys are condition names\n"
            f"    and the inner dicts have {_REQUIRED_KEYS} as keys\n"
        )

    def setup(self, trainer, solver, stage):
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
        conditions = solver.problem.conditions

        # expand single normalizer to all conditions
        if self.normalizer is not None:
            if set(self.normalizer.keys()) == _REQUIRED_KEYS:
                self.normalizer = {c: self.normalizer for c in conditions}

            # check condition keys
            for cond in self.normalizer:
                if cond not in conditions:
                    raise RuntimeError(
                        f"Condition '{cond}' not found in the normalizer dict. "
                        f"Got {list(self.normalizer)}, expected {list(conditions)}."
                    )
                if (
                    hasattr(conditions[cond], "equation")
                    and self.apply_to == "target"
                ):
                    raise RuntimeError(
                        f"Condition '{cond}' contains an equation object, "
                        "so there is no available target data to scale."
                    )

        # select dataset and normalize
        stage = stage or "fit"
        if stage == "fit" and self.stage in ["train", "all"]:
            if self.normalizer is None:
                self.normalizer = {}
                self._compute_scale_shift(
                    conditions, trainer.data_module.train_dataset
                )
            self._scale_data(trainer.data_module.train_dataset)
        if stage == "fit" and self.stage in ["validate", "all"]:
            if self.normalizer is None:
                raise RuntimeError(
                    "Cannot compute scale and shift from test data. "
                    "Please provide a valid normalizer dict."
                )
            self._scale_data(trainer.data_module.val_dataset)
        if stage == "test" and self.stage in ["test", "all"]:
            if self.normalizer is None:
                raise RuntimeError(
                    "Cannot compute scale and shift from test data. "
                    "Please provide a valid normalizer dict."
                )
            self._scale_data(trainer.data_module.test_dataset)

        return super().setup(trainer, solver, stage)

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
                self.normalizer[cond] = {
                    "shift": data.mean(dim=0),
                    "scale": data.std(dim=0) + 1e-8,
                }

    @staticmethod
    def scale_fn(value, scale, shift):
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
        if isinstance(value, LabelTensor):
            return LabelTensor((value.tensor - shift) / scale, value.labels)
        return (value - shift) / scale

    def _scale_data(self, dataset):
        """
        Apply normalization to a dataset in-place.

        :param dataset: Dataset object with `conditions_dict` and `update_data`.
        :type dataset: object
        """
        new_points = {}
        for cond in self.normalizer:
            current_points = dataset.conditions_dict[cond][self.apply_to]
            scale = self.normalizer[cond]["scale"]
            shift = self.normalizer[cond]["shift"]
            new_points[cond] = {
                self.apply_to: self.scale_fn(current_points, scale, shift)
            }
        dataset.update_data(new_points)
