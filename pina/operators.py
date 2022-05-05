"""Module for operators vectorize implementation"""
import torch

from pina.label_tensor import LabelTensor


def grad(output_, input_):
    """
    TODO
    """
    if not isinstance(input_, LabelTensor):
        raise TypeError

    gradients = torch.autograd.grad(
        output_,
        input_,
        grad_outputs=torch.ones(output_.size()).to(
            dtype=input_.dtype,
            device=input_.device),
        create_graph=True, retain_graph=True, allow_unused=True)[0]
    return LabelTensor(gradients, input_.labels)


def div(output_, input_):
    """
    TODO
    """
    if output_.shape[1] == 1:
        div = grad(output_, input_).sum(axis=1)
    else:  # really to improve
        a = []
        for o in output_.T:
            a.append(grad(o, input_).extract(['x', 'y']))
        div = torch.zeros(output_.shape[0], 1)
        for i in range(output_.shape[1]):
            div += a[i][:, i].reshape(-1, 1)

    return div


def nabla(output_, input_):
    """
    TODO
    """
    return div(grad(output_, input_).extract(['x', 'y']), input_)


def advection_term(output_, input_):
    """
    TODO
    """
    dimension = len(output_.labels)
    for i, label in enumerate(output_.labels):
        # compute u dot gradient in each direction 
        gradient_loc = grad(output_.extract([label]), input_).extract(input_.labels[:dimension])
        dim_0 = gradient_loc.shape[0]
        dim_1 = gradient_loc.shape[1]
        u_dot_grad_loc = torch.bmm(output_.view(dim_0, 1, dim_1),
                             gradient_loc.view(dim_0, dim_1, 1))
        u_dot_grad_loc = LabelTensor(torch.reshape(u_dot_grad_loc,
                                 (u_dot_grad_loc.shape[0], u_dot_grad_loc.shape[1])), [input_.labels[i]])
        if i==0:
            adv_term = u_dot_grad_loc
        else:
            adv_term = adv_term.append(u_dot_grad_loc)
    return adv_term
