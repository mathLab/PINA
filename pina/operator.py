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

    def _scalar_grad(output_, input_, d):
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
            outputs=output_,
            inputs=input_,
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
    d = d if isinstance(d, list) else [d]
    components = components if isinstance(components, list) else [components]

    # Check if all labels are present in the input tensor
    if not all(di in input_.labels for di in d):
        raise RuntimeError("Derivative labels missing from input tensor.")

    # Check if all labels are present in the output tensor
    if not all(c in output_.labels for c in components):
        raise RuntimeError("Component label missing from output tensor.")

    # Scalar gradient
    if output_.shape[1] == 1:
        return LabelTensor(
            _scalar_grad(output_=output_, input_=input_, d=d),
            labels=[f"d{output_.labels[0]}d{i}" for i in d],
        )

    # Vector gradient
    grads = torch.cat(
        [
            _scalar_grad(output_=output_.extract(c), input_=input_, d=d)
            for c in components
        ],
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
    :param components: The names of the output variables for which to compute
        the divergence. It must be a subset of the output labels.
        If ``None``, all output variables are considered. Default is ``None``.
    :type components: str | list[str]
    :param d: The names of the input variables with respect to which the
        divergence is computed. It must be a subset of the input labels.
        If ``None``, all input variables are considered. Default is ``None``.
    :type components: str | list[str]
    :raises TypeError: If the input tensor is not a LabelTensor.
    :raises ValueError: If the length of ``components`` and ``d`` do not match.
    :return: The computed divergence tensor.
    :rtype: LabelTensor
    """
    # Check if the input is a LabelTensor
    if not isinstance(input_, LabelTensor):
        raise TypeError("Input must be a LabelTensor.")

    # If no labels are provided, use all labels
    d = d or input_.labels
    components = components or output_.labels

    # Convert to list if not already
    d = d if isinstance(d, list) else [d]
    components = components if isinstance(components, list) else [components]

    # Components and d must be of the same length
    if len(components) != len(d):
        raise ValueError(
            "Divergence requires components and d to be of the same length."
        )

    grad_out = grad(output_=output_, input_=input_, components=components, d=d)
    tensors_to_sum = [
        grad_out.extract(f"d{c}d{d_}") for c, d_ in zip(components, d)
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
    :param str method: The method used to compute the Laplacian. Available
        methods are ``std`` and ``divgrad``. The ``std`` method computes the
        trace of the Hessian matrix, while the ``divgrad`` method computes the
        divergence of the gradient. Default is ``std``.
    :raises TypeError: If the input tensor is not a LabelTensor.
    :raises ValueError: If the passed method is neither ``std`` nor ``divgrad``.
    :return: The computed laplacian tensor.
    :rtype: LabelTensor
    """

    def _scalar_laplacian(output_, input_, d):
        """
        Compute the laplacian of a scalar-valued ``output_``.

        :param LabelTensor output_: The output tensor on which the laplacian is
            computed. It must be a column tensor.
        :param LabelTensor input_: The input tensor with respect to which the
            laplacian is computed.
        :param list[str] d: The names of the input variables with respect to
            which the laplacian is computed. It must be a subset of the input
            labels. If ``None``, all input variables are considered.
        :return: The computed laplacian tensor.
        :rtype: LabelTensor
        """
        first_grad = grad(output_=output_, input_=input_, d=d)
        second_grad = grad(output_=first_grad, input_=input_, d=d)
        return torch.sum(second_grad, dim=1, keepdim=True)

    # Check if the input is a LabelTensor
    if not isinstance(input_, LabelTensor):
        raise TypeError("Input must be a LabelTensor.")

    # If no labels are provided, use all labels
    d = d or input_.labels
    components = components or output_.labels

    # Convert to list if not already
    d = d if isinstance(d, list) else [d]
    components = components if isinstance(components, list) else [components]

    # Scalar laplacian
    if output_.shape[1] == 1:
        return LabelTensor(
            _scalar_laplacian(output_=output_, input_=input_, d=d),
            labels=[f"dd{c}" for c in components],
        )

    # Initialize the result tensor and its labels
    labels = [f"dd{c}" for c in components]
    result = torch.empty(
        input_.shape[0], len(components), device=output_.device
    )

    # Vector laplacian
    if method == "std":
        result = torch.stack(
            [
                _scalar_laplacian(
                    output_=output_.extract(c), input_=input_, d=d
                ).flatten()
                for c in components
            ],
            dim=1,
        )

    elif method == "divgrad":
        grads = grad(output_=output_, input_=input_, components=components, d=d)
        result = torch.stack(
            [
                div(
                    output_=grads,
                    input_=input_,
                    components=[f"d{c}d{i}" for i in d],
                    d=d,
                ).flatten()
                for c in components
            ],
            dim=1,
        )

    else:
        raise ValueError(
            "Invalid method. Available methods are ``std`` and ``divgrad``."
        )

    return LabelTensor(result, labels=labels)


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
    :param components: The names of the output variables for which to compute
        the advection. It must be a subset of the output labels.
        If ``None``, all output variables are considered. Default is ``None``.
    :type components: str | list[str]
    :param d: The names of the input variables with respect to which the
        advection is computed. It must be a subset of the input labels.
        If ``None``, all input variables are considered. Default is ``None``.
    :type d: str | list[str]
    :raises TypeError: If the input tensor is not a LabelTensor.
    :raises RuntimeError: If the velocity field is not in the output labels.
    :return: The computed advection tensor.
    :rtype: torch.Tensor
    """
    # Check if the input is a LabelTensor
    if not isinstance(input_, LabelTensor):
        raise TypeError("Input must be a LabelTensor.")

    # If no labels are provided, use all labels
    d = d or input_.labels
    components = components or output_.labels

    # Convert to list if not already
    d = d if isinstance(d, list) else [d]
    components = components if isinstance(components, list) else [components]

    # Check if velocity field is present in the output labels
    if velocity_field not in output_.labels:
        raise RuntimeError(
            f"Velocity {velocity_field} is not present in the output labels."
        )

    # Save the velocity field
    velocity = output_.extract(velocity_field)

    # Remove the velocity field from the components
    filter_components = [c for c in components if c != velocity_field]

    tmp = (
        grad(output_=output_, input_=input_, components=filter_components, d=d)
        .reshape(-1, len(filter_components), len(d))
        .transpose(0, 1)
    )

    return (tmp * velocity).sum(dim=2).T
