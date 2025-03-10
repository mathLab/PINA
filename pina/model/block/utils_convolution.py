"""
TODO
"""

import torch


def check_point(x, current_stride, dim):
    """
    TODO
    """
    max_stride = current_stride + dim
    indeces = torch.logical_and(
        x[..., :-1] < max_stride, x[..., :-1] >= current_stride
    ).all(dim=-1)
    return indeces


def map_points_(x, filter_position):
    """Mapping function n dimensional case

    :param x: input data of two dimension
    :type x: torch.tensor
    :param filter_position: position of the filter
    :type dim: list[numeric]
    :return: data mapped inplace
    :rtype: torch.tensor
    """
    x.add_(-filter_position)

    return x


def optimizing(f):
    """Decorator for calling a function just once

    :param f: python function
    :type f: function
    """

    def wrapper(*args, **kwargs):

        if kwargs["type_"] == "forward":
            if not wrapper.has_run_inverse:
                wrapper.has_run_inverse = True
                return f(*args, **kwargs)

        if kwargs["type_"] == "inverse":
            if not wrapper.has_run:
                wrapper.has_run = True
                return f(*args, **kwargs)

        return f(*args, **kwargs)

    wrapper.has_run_inverse = False
    wrapper.has_run = False

    return wrapper
