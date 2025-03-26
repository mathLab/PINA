"""
Module for vectorized differential operators implementation.

Differential operators are used to define differential problems and are
implemented to run efficiently on various accelerators, including CPU, GPU, TPU,
and MPS.

Each differential operator takes the following inputs:
- A tensor on which the operator is applied.
- A tensor with respect to which the operator is computed.
- The names of the output variables for which the operator is evaluated.
- The names of the variables with respect to which the operator is computed.
"""

import torch
from pina.label_tensor import LabelTensor


def grad(output_, input_, components=None, d=None):
    """
    Compute the gradient of the ``output_`` with respect to the ``input``.

    This operator supports both vector-valued and scalar-valued functions with
    one or multiple input coordinates.

    :param LabelTensor output_: The output tensor on which the gradient is
        computed.
    :param LabelTensor input_: The input tensor with respect to which the
        gradient is computed.
    :param components: The names of the output variables for which to
        compute the gradient. It must be a subset of the output labels.
        If ``None``, all output variables are considered. Default is ``None``.
    :type components: str | list[str]
    :param d: The names of the input variables with respect to which
        the gradient is computed. It must be a subset of the input labels.
        If ``None``, all input variables are considered. Default is ``None``.
    :type d: str | list[str]
    :raises TypeError: If the input tensor is not a LabelTensor.
    :raises RuntimeError: If the output is a scalar field and the components
        are not equal to the output labels.
    :raises NotImplementedError: If the output is neither a vector field nor a
        scalar field.
    :return: The computed gradient tensor.
    :rtype: LabelTensor
    """

    def grad_scalar_output(output_, input_, d):
        """
        Compute the gradient of a scalar-valued ``output_``.

        :param LabelTensor output_: The output tensor on which the gradient is
            computed. It must be a column tensor.
        :param LabelTensor input_: The input tensor with respect to which the
            gradient is computed.
        :param d: The names of the input variables with respect to
            which the gradient is computed. It must be a subset of the input
            labels. If ``None``, all input variables are considered.
        :type d: str | list[str]
        :raises RuntimeError: If a vectorial function is passed.
        :raises RuntimeError: If missing derivative labels.
        :return: The computed gradient tensor.
        :rtype: LabelTensor
        """

        if len(output_.labels) != 1:
            raise RuntimeError("only scalar function can be differentiated")
        if not all(di in input_.labels for di in d):
            raise RuntimeError("derivative labels missing from input tensor")

        output_fieldname = output_.labels[0]
        gradients = torch.autograd.grad(
            output_,
            input_,
            grad_outputs=torch.ones(
                output_.size(), dtype=output_.dtype, device=output_.device
            ),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        gradients.labels = input_.stored_labels
        gradients = gradients[..., [input_.labels.index(i) for i in d]]
        gradients.labels = [f"d{output_fieldname}d{i}" for i in d]
        return gradients

    if not isinstance(input_, LabelTensor):
        raise TypeError

    if d is None:
        d = input_.labels

    if components is None:
        components = output_.labels

    if not isinstance(components, list):
        components = [components]

    if not isinstance(d, list):
        d = [d]

    if output_.shape[1] == 1:  # scalar output ################################

        if components != output_.labels:
            raise RuntimeError
        gradients = grad_scalar_output(output_, input_, d)

    elif (
        output_.shape[output_.ndim - 1] >= 2
    ):  # vector output ##############################
        tensor_to_cat = []
        for i, c in enumerate(components):
            c_output = output_.extract([c])
            tensor_to_cat.append(grad_scalar_output(c_output, input_, d))
        gradients = LabelTensor.cat(tensor_to_cat, dim=output_.tensor.ndim - 1)
    else:
        raise NotImplementedError

    return gradients


def div(output_, input_, components=None, d=None):
    """
    Compute the divergence of the ``output_`` with respect to ``input``.

    This operator supports vector-valued functions with multiple input
    coordinates.

    :param LabelTensor output_: The output tensor on which the divergence is
        computed.
    :param LabelTensor input_: The input tensor with respect to which the
        divergence is computed.
    :param components: The names of the output variables for which to
        compute the divergence. It must be a subset of the output labels.
        If ``None``, all output variables are considered. Default is ``None``.
    :type components: str | list[str]
    :param d: The names of the input variables with respect to which
        the divergence is computed. It must be a subset of the input labels.
        If ``None``, all input variables are considered. Default is ``None``.
    :type d: str | list[str]
    :raises TypeError: If the input tensor is not a LabelTensor.
    :raises ValueError: If the output is a scalar field.
    :raises ValueError: If the number of components is not equal to the number
        of input variables.
    :return: The computed divergence tensor.
    :rtype: LabelTensor
    """
    if not isinstance(input_, LabelTensor):
        raise TypeError

    if d is None:
        d = input_.labels

    if components is None:
        components = output_.labels

    if not isinstance(components, list):
        components = [components]

    if not isinstance(d, list):
        d = [d]

    if output_.shape[1] < 2 or len(components) < 2:
        raise ValueError("div supported only for vector fields")

    if len(components) != len(d):
        raise ValueError

    grad_output = grad(output_, input_, components, d)
    labels = [None] * len(components)
    tensors_to_sum = []
    for i, (c, d_) in enumerate(zip(components, d)):
        c_fields = f"d{c}d{d_}"
        tensors_to_sum.append(grad_output.extract(c_fields))
        labels[i] = c_fields
    div_result = LabelTensor.summation(tensors_to_sum)
    return div_result


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
