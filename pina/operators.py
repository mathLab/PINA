"""Module for operators vectorize implementation"""
import torch

from pina.label_tensor import LabelTensor


def grad(output_, input_, components=None, d=None):
    """
    TODO
    """

    def grad_scalar_output(output_, input_, d):
        """
        """

        if len(output_.labels) != 1:
            raise RuntimeError
        if not all([di in input_.labels for di in d]):
            raise RuntimeError

        output_fieldname = output_.labels[0]
        gradients = torch.autograd.grad(
            output_,
            input_,
            grad_outputs=torch.ones(output_.size(), dtype=output_.dtype,
                                    device=output_.device),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]

        gradients.labels = input_.labels
        gradients = gradients.extract(d)
        gradients.labels = [f'd{output_fieldname}d{i}' for i in d]

        return gradients

    if not isinstance(input_, LabelTensor):
        raise TypeError

    if d is None:
        d = input_.labels

    if components is None:
        components = output_.labels

    if output_.shape[1] == 1:  # scalar output ################################

        if components != output_.labels:
            raise RuntimeError
        gradients = grad_scalar_output(output_, input_, d)

    elif output_.shape[1] >= 2:  # vector output ##############################

        for i, c in enumerate(components):
            c_output = output_.extract([c])
            if i == 0:
                gradients = grad_scalar_output(c_output, input_, d)
            else:
                gradients = gradients.append(
                    grad_scalar_output(c_output, input_, d))
    else:
        raise NotImplementedError

    return gradients


def div(output_, input_, components=None, d=None):
    """
    TODO
    """
    if not isinstance(input_, LabelTensor):
        raise TypeError

    if d is None:
        d = input_.labels

    if components is None:
        components = output_.labels

    if output_.shape[1] < 2 or len(components) < 2:
        raise ValueError('div supported only for vector field')

    if len(components) != len(d):
        raise ValueError

    grad_output = grad(output_, input_, components, d)
    div = torch.zeros(input_.shape[0], 1, device=output_.device)
    labels = [None] * len(components)
    for i, (c, d) in enumerate(zip(components, d)):
        c_fields = f'd{c}d{d}'
        div[:, 0] += grad_output.extract(c_fields).sum(axis=1)
        labels[i] = c_fields

    div = div.as_subclass(LabelTensor)
    div.labels = ['+'.join(labels)]
    return div


def nabla(output_, input_, components=None, d=None, method='std'):
    """
    TODO
    """
    if d is None:
        d = input_.labels

    if components is None:
        components = output_.labels

    if len(components) != len(d) and len(components) != 1:
        raise ValueError

    if method == 'divgrad':
        raise NotImplementedError
        # TODO fix
        # grad_output = grad(output_, input_, components, d)
        # result = div(grad_output, input_, d=d)
    elif method == 'std':

        if len(components) == 1:
            grad_output = grad(output_, input_, components=components, d=d)
            result = torch.zeros(output_.shape[0], 1, device=output_.device)
            for i, label in enumerate(grad_output.labels):
                gg = grad(grad_output, input_, d=d, components=[label])
                result[:, 0] += gg[:, i]
            labels = [f'dd{components[0]}']

        else:
            result = torch.empty(input_.shape[0], len(components),
                                 device=output_.device)
            labels = [None] * len(components)
            for idx, (ci, di) in enumerate(zip(components, d)):

                if not isinstance(ci, list):
                    ci = [ci]
                if not isinstance(di, list):
                    di = [di]

                grad_output = grad(output_, input_, components=ci, d=di)
                result[:, idx] = grad(grad_output, input_, d=di).flatten()
                labels[idx] = f'dd{ci}dd{di}'

    result = result.as_subclass(LabelTensor)
    result.labels = labels
    return result


def advection(output_, input_, velocity_field, components=None, d=None):
    if d is None:
        d = input_.labels

    if components is None:
        components = output_.labels

    tmp = grad(output_, input_, components, d
               ).reshape(-1, len(components), len(d)).transpose(0, 1)

    tmp *= output_.extract(velocity_field)
    return tmp.sum(dim=2).T
