import torch
from .condition_interface import ConditionInterface
from ..loss import TimeWeightingInterface, ConstantTimeWeighting
from ..utils import check_consistency


class AutoregressiveCondition(ConditionInterface):
    """
    A specialized condition for autoregressive tasks.
    It generates input/unroll pairs from a single time-series tensor.
    """

    __slots__ = ["input", "unroll"]

    def __init__(
        self,
        data,
        unroll_length,
        num_unrolls=None,
        randomize=True,
        time_weighting=None,
    ):
        """
        Create an AutoregressiveCondition.
        """
        super().__init__()

        self._n_timesteps, n_features = data.shape
        self._unroll_length = unroll_length
        self._requested_num_unrolls = num_unrolls
        self._randomize = randomize

        # time weighting: weight the loss differently along the unroll
        if time_weighting is None:
            self._time_weighting = ConstantTimeWeighting()
        else:
            check_consistency(time_weighting, TimeWeightingInterface)
            self._time_weighting = time_weighting

        # windows creation
        initial_data = []
        unroll_data = []

        for starting_index in self.starting_indices:
            initial_data.append(data[starting_index])
            target_start = starting_index + 1
            unroll_data.append(
                data[target_start : target_start + self._unroll_length, :]
            )

        self.input = torch.stack(initial_data)  # [num_unrolls, features]
        self.unroll = torch.stack(
            unroll_data
        )  # [num_unrolls, unroll_length, features]

    @property
    def unroll_length(self):
        return self._unroll_length

    @property
    def time_weighting(self):
        return self._time_weighting

    @property
    def max_start_idx(self):
        max_start_idx = self._n_timesteps - self._unroll_length
        assert max_start_idx > 0, "Provided data sequence too short"
        return max_start_idx

    @property
    def num_unrolls(self):
        if self._requested_num_unrolls is None:
            return self.max_start_idx
        else:
            assert (
                self._requested_num_unrolls < self.max_start_idx
            ), "too many samples requested"
            return self._requested_num_unrolls

    @property
    def starting_indices(self):
        all_starting_indices = torch.arange(self.max_start_idx)

        if self._randomize:
            perm = torch.randperm(len(all_starting_indices))
            return all_starting_indices[perm[: self.num_unrolls]]
        else:
            selected_indices = torch.linspace(
                0, len(all_starting_indices) - 1, self.num_unrolls
            ).long()
            return all_starting_indices[selected_indices]
