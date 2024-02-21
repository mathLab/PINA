import torch
from pina.utils import check_consistency
from pina import LabelTensor


class FourierEmbedding(torch.nn.Module):
    """
    Fourier Embedding for imposing hard constraint periodic boundary conditions.

    The following
    """
    def __init__(self, input_dimension, periods, output_dimension=None):
        """
        :param int input_dimension: The dimension of the input tensor, it can
            be checked with `tensor.ndim` method.
        :param float | int | dict periods: The periodicity in each dimension for
            the input data. If `float` or `int` is passed, the period is assumed
            constant for all the dimensions of the data. If a `dict` is passed
            the `dict.values` represent periods, while the `dict.keys` represent
            the dimension where the periodicity is applied. The `dict.keys`
            can either be `int` if working with `torch.Tensors` or `str` if
            working with `LabelTensor`.
        :param int output_dimension: The dimension of the output after the
            fourier embedding. If not `None` a `torch.nn.Linear` layer is
            applied to the fourier embedding output to match the desired
            dimensionality, default `None`.
        """
        super().__init__()

        # check input consistency
        check_consistency(periods, (float, int, dict))
        check_consistency(input_dimension, int)
        if output_dimension is not None:
            check_consistency(output_dimension, int)
            self.layer = torch.nn.Linear(input_dimension * 3, output_dimension)
        else:
            self.layer = torch.nn.Identity()

        # checks on the periods
        if isinstance(periods, dict):
            if not all(isinstance(dim, (str, int)) and 
                       isinstance(period, (float, int)) 
                       for dim, period in periods.items()):
                raise TypeError('In dictionary periods, keys must be integers'
                                ' or strings, and values must be float or int.')
            self._period = periods
        else:
            self._period = {k: periods for k in range(input_dimension)}


    def forward(self, x):
        """_summary_

        :param torch.Tensor x: Input tensor.
        :return: Fourier embeddings of the input.
        :rtype: torch.Tensor
        """
        self._omega = torch.stack([torch.pi * 2. / torch.tensor([val]) 
                                   for val in self._period.values()], dim=-1)
        x = self._get_vars(x, list(self._period.keys()))
        return self.layer(torch.cat([torch.ones_like(x),
                          torch.cos(self._omega * x),
                          torch.sin(self._omega * x)], dim=-1))

    def _get_vars(self, x, indeces):
        """
        Get variables from input tensor ordered by specific indeces.

        :param torch.Tensor x: The input tensor to extract.
        :param list[int] | list[str] indeces: List of indeces to extract.
        :return: The extracted tensor given the indeces.
        :rtype: torch.Tensor
        """
        if isinstance(indeces[0], str):
            try:
                return x.extract(indeces)
            except AttributeError:
                raise RuntimeError(
                    'Not possible to extract input variables from tensor.'
                    ' Ensure that the passed tensor is a LabelTensor or'
                    ' pass list of integers to extract variables. For'
                    ' more information refer to warning in the documentation.')
        elif isinstance(indeces[0], int):
            return x[..., indeces]
        else:
            raise RuntimeError(
                'Not able to extract right indeces for tensor.'
                ' For more information refer to warning in the documentation.')
        
    @property
    def period(self):
        """
        The period of the periodic function to approximate.
        """
        return 2 * torch.pi / self._omega
    
    
def grad(u, x):
    """
    Compute the first derivative of u with respect to x.

    Parameters:
    - u (torch.Tensor): The tensor for which the derivative is calculated (requires_grad=True).
    - x (torch.Tensor): The tensor with respect to which the derivative is calculated.

    Returns:
    - torch.Tensor: The first derivative of u with respect to x.
    """
    # Calculate the gradient
    return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True, retain_graph=True)[0]

foo_func = torch.nn.Sequential(torch.nn.Linear(1, 10),
                               torch.nn.Tanh(),
                               torch.nn.Linear(10, 10),
                               torch.nn.Tanh(),
                               torch.nn.Linear(10, 1))
periodic_foo_func = torch.nn.Sequential(FourierEmbedding(input_dimension=1, periods=1, output_dimension=10),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear(10, 10),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear(10, 1))
data = torch.linspace(0, 1, 100).reshape(-1, 1)
data_1 = 1. + data
data_2 = 2. + data
data_10 = 10. + data
data_100 = 100. + data
print(data.ndim)
data.requires_grad = True
data_1.requires_grad = True
data_2.requires_grad = True
data_10.requires_grad = True
data_100.requires_grad = True

# check periodicity
output_foo_func = foo_func(data)
output_foo_func_x = grad(output_foo_func, data)
output_foo_func_xx = grad(output_foo_func_x, data)
output_foo_func_xxx = grad(output_foo_func_xx, data)
output_foo_func_xxxx = grad(output_foo_func_xxx, data)
for idx, data_new in zip([1, 2, 100], [data_1, data_2, data_100]):
    output_foo_func_L = foo_func(data_new)
    output_foo_func_L_x = grad(output_foo_func_L, data_new)
    output_foo_func_L_xx = grad(output_foo_func_L_x, data_new)
    output_foo_func_L_xxx = grad(output_foo_func_L_xx, data_new)
    output_foo_func_L_xxxx = grad(output_foo_func_L_xxx, data_new)
    print(f'Foo func u(x)=u(x+{idx})', torch.nn.functional.l1_loss(output_foo_func, output_foo_func_L))
    print(f'Foo func Du(x)=Du(x+{idx})', torch.nn.functional.l1_loss(output_foo_func_x, output_foo_func_L_x))
    print(f'Foo func D2u(x)=D2u(x+{idx})', torch.nn.functional.l1_loss(output_foo_func_xx, output_foo_func_L_xx))
    print(f'Foo func D3u(x)=D3u(x+{idx})', torch.nn.functional.l1_loss(output_foo_func_xxx, output_foo_func_L_xxx))
    print(f'Foo func D4u(x)=D4u(x+{idx})', torch.nn.functional.l1_loss(output_foo_func_xxxx, output_foo_func_L_xxxx))

print()
# check periodicity
periodic_output_foo_func = periodic_foo_func(data)
periodic_output_foo_func_x = grad(periodic_output_foo_func, data)
periodic_output_foo_func_xx = grad(periodic_output_foo_func_x, data)
periodic_output_foo_func_xxx = grad(periodic_output_foo_func_xx, data)
periodic_output_foo_func_xxxx = grad(periodic_output_foo_func_xxx, data)
for idx, data_new in zip([1, 2, 100], [data_1, data_2, data_100]):
    periodic_output_foo_func_L = periodic_foo_func(data_new)
    periodic_output_foo_func_L_x = grad(periodic_output_foo_func_L, data_new)
    periodic_output_foo_func_L_xx = grad(periodic_output_foo_func_L_x, data_new)
    periodic_output_foo_func_L_xxx = grad(periodic_output_foo_func_L_xx, data_new)
    periodic_output_foo_func_L_xxxx = grad(periodic_output_foo_func_L_xxx, data_new)
    print(f'Periodic func u(x)=u(x+{idx})', torch.nn.functional.l1_loss(periodic_output_foo_func, periodic_output_foo_func_L))
    print(f'Periodic func Du(x)=Du(x+{idx})', torch.nn.functional.l1_loss(periodic_output_foo_func_x, periodic_output_foo_func_L_x))
    print(f'Periodic func D2u(x)=D2u(x+{idx})', torch.nn.functional.l1_loss(periodic_output_foo_func_xx, periodic_output_foo_func_L_xx))
    print(f'Periodic func D3u(x)=D3u(x+{idx})', torch.nn.functional.l1_loss(periodic_output_foo_func_L_xxx, periodic_output_foo_func_xxx))
    print(f'Periodic func D4u(x)=D4u(x+{idx})', torch.nn.functional.l1_loss(periodic_output_foo_func_L_xxxx, periodic_output_foo_func_xxxx))

