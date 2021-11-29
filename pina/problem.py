import torch

class Problem(object):

    def __init__(self, *args, **kwargs):
        raise NotImplemented

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, variables):
        if variables is None:
            variables = ['var']
        self._variables = variables

    @property
    def boundary_conditions(self):
        return self._bc

    @boundary_conditions.setter
    def boundary_conditions(self, bc):
        if isinstance(bc, (list, tuple)):
            bc = {'var': bc}
        self._bc = bc

    @property
    def spatial_dimensions(self):
        return self._spatial_dimensions

    @staticmethod
    def grad(output_, input_):
        gradients = torch.autograd.grad(
            output_, 
            input_.tensor, 
            grad_outputs=torch.ones(output_.size()).to(
                dtype=input_.tensor.dtype, 
                device=input_.tensor.device), 
            create_graph=True, retain_graph=True, allow_unused=True)[0]
        from pina.label_tensor import LabelTensor
        return LabelTensor(gradients, input_.labels)



    def __str__(self):
        s = ''
        #s = 'Variables: {}\n'.format(self.variables)
        return s
