"""
Module for vectorized differential operators implementation.

Differential operators are used to define differential problems and are
implemented to run efficiently on various accelerators, including CPU, GPU, and
TPU.

Each differential operator takes the following inputs:
- A tensor on which the operator is applied.
- A tensor with respect to which the operator is computed.
- The names of the output variables for which the operator is evaluated.
- The names of the variables with respect to which the operator is computed.
"""

import torch
from pina import LabelTensor


def grad(output_, input_, components=None, d=None):
    """
    Compute the gradient of the ``output_`` with respect to the ``input``.

    This operator supports both vector-valued and scalar-valued functions with
    one or multiple input coordinates.

    :param LabelTensor output_: The output tensor on which the gradient is
        computed.
    :param LabelTensor input_: The input tensor with respect to which the
        gradient is computed.
    :param components: The names of the output variables for which to compute
        the gradient. It must be a subset of the output labels.
        If ``None``, all output variables are considered. Default is ``None``.
    :type components: str | list[str]
    :param d: The names of the input variables with respect to which the
        gradient is computed. It must be a subset of the input labels.
        If ``None``, all input variables are considered. Default is ``None``.
    :type d: str | list[str]
    :raises TypeError: If the input tensor is not a LabelTensor.
    :raises RuntimeError: If derivative labels are missing from the ``input_``.
    :raises RUntimeError: If component labels are missing from the ``output_``.
    :return: The computed gradient tensor.
    :rtype: LabelTensor
    """

    def _grad_scalar(output_, input_, d):
        """
        Compute the gradient of a scalar-valued ``output_``.

        :param LabelTensor output_: The output tensor on which the gradient is
            computed. It must be a column tensor.
        :param LabelTensor input_: The input tensor with respect to which the
            gradient is computed.
        :param list[str] d: The names of the input variables with respect to
            which the gradient is computed. It must be a subset of the input
            labels. If ``None``, all input variables are considered.
        :return: The computed gradient tensor.
        :rtype: LabelTensor
        """
        grad_out = torch.autograd.grad(
            output_,
            input_,
            grad_outputs=torch.ones_like(output_),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        return grad_out[..., [input_.labels.index(i) for i in d]]

    if not isinstance(input_, LabelTensor):
        raise TypeError("Input must be a LabelTensor.")

    # If no labels are provided, use all labels
    d = d or input_.labels
    components = components or output_.labels

    # Convert to list if not already
    d = [d] if not isinstance(d, list) else d
    components = (
        [components] if not isinstance(components, list) else components
    )

    if not all(di in input_.labels for di in d):
        raise RuntimeError("Derivative labels missing from input tensor.")

    if not all(c in output_.labels for c in components):
        raise RuntimeError("Component label missing from output tensor.")

    # Scalar gradient
    if output_.shape[1] == 1:
        return LabelTensor(
            _grad_scalar(output_, input_, d),
            labels=[f"d{output_.labels[0]}d{i}" for i in d],
        )

    # Vector gradient
    grads = torch.cat(
        [_grad_scalar(output_.extract([c]), input_, d) for c in components],
        dim=output_.tensor.ndim - 1,
    )

    return LabelTensor(
        grads, labels=[f"d{c}d{i}" for c in components for i in d]
    )


def div(output_, input_, components=None, d=None):
    """
    Compute the divergence of the ``output_`` with respect to ``input``.

    This operator supports vector-valued functions with multiple input
    coordinates.

    :param LabelTensor output_: The output tensor on which the divergence is
        computed.
    :param LabelTensor input_: The input tensor with respect to which the
        divergence is computed.
    :param list[str] components: The names of the output variables for which to
        compute the divergence. It must be a subset of the output labels.
        If ``None``, all output variables are considered. Default is ``None``.
    :param list[str] d: The names of the input variables with respect to which
        the divergence is computed. It must be a subset of the input labels.
        If ``None``, all input variables are considered. Default is ``None``.
    :raises TypeError: If the input tensor is not a LabelTensor.
    :raises ValueError: If the output tensor is a scalar field.
    :raises ValueError: If the length of ``components`` and ``d`` do not match.
    :return: The computed divergence tensor.
    :rtype: LabelTensor
    """
    if not isinstance(input_, LabelTensor):
        raise TypeError("Input must be a LabelTensor.")

    # If no labels are provided, use all labels
    d = d or input_.labels
    components = components or output_.labels

    # Convert to list if not already
    d = [d] if not isinstance(d, list) else d
    components = (
        [components] if not isinstance(components, list) else components
    )

    # Raise error for scalar valued output
    if len(components) < 2:
        raise ValueError("Divergence is supported only for vector fields")

    # Components and d must be of the same length
    if len(components) != len(d):
        raise ValueError(
            "Divergence requires components and d to be of the same length."
        )

    grad_output = grad(output_, input_, components, d)
    tensors_to_sum = [
        grad_output.extract(f"d{c}d{d_}") for c, d_ in zip(components, d)
    ]

    return LabelTensor.summation(tensors_to_sum)


def laplacian(output_, input_, components=None, d=None, method="std"):
    """
    Compute the laplacian of the ``output_`` with respect to ``input``.

    This operator supports both vector-valued and scalar-valued functions with
    one or multiple input coordinates.

    :param LabelTensor output_: The output tensor on which the laplacian is
        computed.
    :param LabelTensor input_: The input tensor with respect to which the
        laplacian is computed.
    :param components: The names of the output variables for which to
        compute the laplacian. It must be a subset of the output labels.
        If ``None``, all output variables are considered. Default is ``None``.
    :type components: str | list[str]
    :param d: The names of the input variables with respect to which
        the laplacian is computed. It must be a subset of the input labels.
        If ``None``, all input variables are considered. Default is ``None``.
    :type d: str | list[str]
    :param str method: The method used to compute the Laplacian. Default is
        ``std``.
    :raises NotImplementedError: If ``std=divgrad``.
    :return: The computed laplacian tensor.
    :rtype: LabelTensor
    """

    def scalar_laplace(output_, input_, components, d):
        """
        Compute the laplacian of a scalar-valued ``output_``.

        :param LabelTensor output_: The output tensor on which the laplacian is
            computed. It must be a column tensor.
        :param LabelTensor input_: The input tensor with respect to which the
            laplacian is computed.
        :param components: The names of the output variables for which
            to compute the laplacian. It must be a subset of the output labels.
            If ``None``, all output variables are considered.
        :type components: str | list[str]
        :param d: The names of the input variables with respect to
            which the laplacian is computed. It must be a subset of the input
            labels. If ``None``, all input variables are considered.
        :type d: str | list[str]
        :return: The computed laplacian tensor.
        :rtype: LabelTensor
        """

        grad_output = grad(output_, input_, components=components, d=d)
        result = torch.zeros(output_.shape[0], 1, device=output_.device)

        for i, label in enumerate(grad_output.labels):
            gg = grad(grad_output, input_, d=d, components=[label])
            result[:, 0] += super(torch.Tensor, gg.T).__getitem__(i)

        return result

    if d is None:
        d = input_.labels

    if components is None:
        components = output_.labels

    if not isinstance(components, list):
        components = [components]

    if not isinstance(d, list):
        d = [d]

    if method == "divgrad":
        raise NotImplementedError("divgrad not implemented as method")

    if method == "std":

        result = torch.empty(
            input_.shape[0], len(components), device=output_.device
        )
        labels = [None] * len(components)
        for idx, c in enumerate(components):
            result[:, idx] = scalar_laplace(output_, input_, [c], d).flatten()
            labels[idx] = f"dd{c}"

        result = result.as_subclass(LabelTensor)
        result.labels = labels

    return result


def advection(output_, input_, velocity_field, components=None, d=None):
    """
    Perform the advection operation on the ``output_`` with respect to the
    ``input``. This operator support vector-valued functions with multiple input
    coordinates.

    :param LabelTensor output_: The output tensor on which the advection is
        computed.
    :param LabelTensor input_: the input tensor with respect to which advection
        is computed.
    :param str velocity_field: The name of the output variable used as velocity
        field. It must be chosen among the output labels.
    :param components: The names of the output variables for which
        to compute the advection. It must be a subset of the output labels.
        If ``None``, all output variables are considered. Default is ``None``.
    :type components: str | list[str]
    :param d: The names of the input variables with respect to which
        the advection is computed. It must be a subset of the input labels.
        If ``None``, all input variables are considered. Default is ``None``.
    :type d: str | list[str]
    :return: The computed advection tensor.
    :rtype: LabelTensor
    """
    if d is None:
        d = input_.labels

    if components is None:
        components = output_.labels

    if not isinstance(components, list):
        components = [components]

    if not isinstance(d, list):
        d = [d]

    tmp = (
        grad(output_, input_, components, d)
        .reshape(-1, len(components), len(d))
        .transpose(0, 1)
    )

    tmp *= output_.extract(velocity_field)
    return tmp.sum(dim=2).T
