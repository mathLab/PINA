"""
TODO: Add description
"""

import torch


class Stride:
    """
    TODO
    """

    def __init__(self, dict_):
        """Stride class for continous convolution

        :param param: type of continuous convolution
        :type param: string
        """

        self._dict_stride = dict_
        self._stride_continuous = None
        self._stride_discrete = self._create_stride_discrete(dict_)

    def _create_stride_discrete(self, my_dict):
        """Creating the list for applying the filter

        :param my_dict: Dictionary with the following arguments:
            domain size, starting position of the filter, jump size
            for the filter and direction of the filter
        :type my_dict: dict
        :raises IndexError: Values in the dict must have all same length
        :raises ValueError: Domain values must be greater than 0
        :raises ValueError: Direction must be either equal to 1, -1 or 0
        :raises IndexError: Direction and jumps must have zero in the same
            index
        :return: list of positions for the filter
        :rtype: list
        :Example:


                >>> stride = {"domain": [4, 4],
                            "start": [-4, 2],
                            "jump": [2, 2],
                            "direction": [1, 1],
                            }
                >>> create_stride(stride)
                [[-4.0, 2.0], [-4.0, 4.0], [-2.0, 2.0], [-2.0, 4.0]]
        """

        # we must check boundaries of the input as well

        domain, start, jumps, direction = my_dict.values()

        # checking

        if not all(len(s) == len(domain) for s in my_dict.values()):
            raise IndexError("values in the dict must have all same length")

        if not all(v >= 0 for v in domain):
            raise ValueError("domain values must be greater than 0")

        if not all(v in (0, -1, 1) for v in direction):
            raise ValueError("direction must be either equal to 1, -1 or 0")

        seq_jumps = [i for i, e in enumerate(jumps) if e == 0]
        seq_direction = [i for i, e in enumerate(direction) if e == 0]

        if seq_direction != seq_jumps:
            raise IndexError(
                "direction and jumps must have zero in the same index"
            )

        if seq_jumps:
            for i in seq_jumps:
                jumps[i] = domain[i]
                direction[i] = 1

        # creating the stride grid
        values_mesh = [
            torch.arange(0, i, step).float() for i, step in zip(domain, jumps)
        ]

        values_mesh = [
            single * dim for single, dim in zip(values_mesh, direction)
        ]

        mesh = torch.meshgrid(values_mesh)
        coordinates_mesh = [x.reshape(-1, 1) for x in mesh]

        stride = torch.cat(coordinates_mesh, dim=1) + torch.tensor(start)

        return stride
