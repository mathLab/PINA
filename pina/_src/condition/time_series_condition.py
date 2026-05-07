"""Module for the TimeSeriesCondition class."""

import torch

from pina._src.data.manager.data_manager import _DataManager
from pina._src.core.label_tensor import LabelTensor
from pina._src.condition.base_condition import BaseCondition


class TimeSeriesCondition(BaseCondition):
    """
    Condition for autoregressive time-series training.

    The condition stores an input tensor containing unroll windows with shape
    ``[trajectories, windows, time_steps, *features]`` and computes the
    autoregressive non-aggregated/aggregated temporal loss inside
    :meth:`evaluate` by recursively applying the solver model over time.
    """

    __fields__ = ["input", "eps", "aggregation_strategy", "kwargs"]
    _avail_input_cls = (torch.Tensor, LabelTensor)

    def __new__(cls, input, eps=None, aggregation_strategy=None, kwargs=None):
        if cls != TimeSeriesCondition:
            return super().__new__(cls)

        if not isinstance(input, cls._avail_input_cls):
            raise ValueError(
                "Invalid input type. Expected one of the following: "
                "torch.Tensor, LabelTensor."
            )

        return super().__new__(cls)

    def store_data(self, **kwargs):
        return _DataManager(input=kwargs.get("input"))

    @property
    def input(self):
        return self.data.input

    @property
    def settings(self):
        return {
            "eps": getattr(self, "_eps", None),
            "aggregation_strategy": getattr(
                self, "_aggregation_strategy", None
            ),
            "kwargs": getattr(self, "_kwargs", {}),
        }

    def __init__(self, input, eps=None, aggregation_strategy=None, kwargs=None):
        super().__init__(input=input)
        self._eps = eps
        self._aggregation_strategy = aggregation_strategy
        self._kwargs = kwargs or {}

    def evaluate(self, batch, solver, loss, condition_name=None):
        input_tensor = batch["input"]

        if input_tensor.dim() < 4:
            raise ValueError(
                "The provided input tensor must have at least 4 dimensions:"
                " [trajectories, windows, time_steps, *features]."
                f" Got shape {input_tensor.shape}."
            )

        current_state = input_tensor[:, :, 0]
        losses = []
        step_kwargs = self._kwargs.copy()

        for step in range(1, input_tensor.shape[2]):
            processed_input = solver.preprocess_step(
                current_state, **step_kwargs
            )
            output = solver.forward(processed_input)
            predicted_state = solver.postprocess_step(output, **step_kwargs)

            target_state = input_tensor[:, :, step]
            step_loss = loss(predicted_state, target_state, **step_kwargs)
            losses.append(step_loss)
            current_state = predicted_state

        step_losses = torch.stack(losses).as_subclass(torch.Tensor)

        with torch.no_grad():
            name = condition_name or getattr(self, "name", None) or "default"
            # weights = solver._get_weights(name, step_losses, self._eps)

        aggregation_strategy = self._aggregation_strategy or torch.mean
        return aggregation_strategy(step_losses)  # * weights)

    @staticmethod
    def unroll(data, unroll_length, n_unrolls=None, randomize=True):
        """
        Create unrolling time windows from temporal data.

        This function takes as input a tensor of shape
        ``[trajectories, time_steps, *features]`` and produces a tensor of
        shape ``[trajectories, windows, unroll_length, *features]``.
        Each window contains a sequence of subsequent states used for
        computing the multi-step loss during training.

        :param data: The temporal data tensor to be unrolled.
        :type data: torch.Tensor | LabelTensor
        :param int unroll_length: The number of time steps in each window.
        :param int n_unrolls: The maximum number of windows to return.
            If ``None``, all valid windows are returned. Default is ``None``.
        :param bool randomize: If ``True``, starting indices are randomly
            permuted before applying ``n_unrolls``. Default is ``True``.
        :raise ValueError: If the input ``data`` has less than 3 dimensions.
        :raise ValueError: If ``unroll_length`` is greater or equal to the
            number of time steps in ``data``.
        :return: A tensor of unrolled windows.
        :rtype: torch.Tensor | LabelTensor
        """
        if data.dim() < 3:
            raise ValueError(
                "The provided data tensor must have at least 3 dimensions:"
                " [trajectories, time_steps, *features]."
                f" Got shape {data.shape}."
            )

        start_idx = TimeSeriesCondition._get_start_idx(
            n_steps=data.shape[1],
            unroll_length=unroll_length,
            n_unrolls=n_unrolls,
            randomize=randomize,
        )

        windows = [data[:, s : s + unroll_length] for s in start_idx]
        return torch.stack(windows, dim=1)

    @staticmethod
    def _get_start_idx(n_steps, unroll_length, n_unrolls=None, randomize=True):
        """
        Determine starting indices for unroll windows.

        :param int n_steps: The total number of time steps in the data.
        :param int unroll_length: The number of time steps in each window.
        :param int n_unrolls: The maximum number of windows to return.
            If ``None``, all valid windows are returned. Default is ``None``.
        :param bool randomize: If ``True``, starting indices are randomly
            permuted before applying ``n_unrolls``. Default is ``True``.
        :raise ValueError: If ``unroll_length`` is greater or equal to the
            number of time steps in ``data``.
        :return: A tensor of starting indices for unroll windows.
        :rtype: torch.Tensor
        """
        last_idx = n_steps - unroll_length

        if last_idx < 0:
            raise ValueError(
                "Cannot create unroll windows: "
                f"unroll_length ({unroll_length})"
                " cannot be greater or equal to the number of time_steps"
                f" ({n_steps})."
            )

        indices = torch.arange(last_idx + 1)

        if randomize:
            indices = indices[torch.randperm(len(indices))]

        if n_unrolls is not None and n_unrolls < len(indices):
            indices = indices[:n_unrolls]

        return indices
