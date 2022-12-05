"""Module for DeepONet model"""
import torch
import torch.nn as nn

from pina import LabelTensor


class DeepONet(torch.nn.Module):
    """
    The PINA implementation of DeepONet network.

    .. seealso::

        **Original reference**: Lu, L., Jin, P., Pang, G. et al. *Learning
        nonlinear operators via DeepONet based on the universal approximation
        theorem of operators*. Nat Mach Intell 3, 218–229 (2021).
        DOI: `10.1038/s42256-021-00302-5
        <https://doi.org/10.1038/s42256-021-00302-5>`_

    """
    def __init__(self, branch_net, trunk_net, output_variables):
        """
        :param torch.nn.Module branch_net: the neural network to use as branch
            model. It has to take as input a :class:`LabelTensor`. The number
            of dimension of the output has to be the same of the `trunk_net`.
        :param torch.nn.Module trunk_net: the neural network to use as trunk
            model. It has to take as input a :class:`LabelTensor`. The number
            of dimension of the output has to be the same of the `branch_net`.
        :param list(str) output_variables: the list containing the labels
            corresponding to the components of the output computed by the
            model.

        :Example:
            >>> branch = FFN(input_variables=['a', 'c'], output_variables=20)
            >>> trunk = FFN(input_variables=['b'], output_variables=20)
            >>> onet = DeepONet(trunk_net=trunk, branch_net=branch
            >>>                 output_variables=output_vars)
            DeepONet(
              (trunk_net): FeedForward(
                (extra_features): Sequential()
                (model): Sequential(
                (0): Linear(in_features=1, out_features=20, bias=True)
                (1): Tanh()
                (2): Linear(in_features=20, out_features=20, bias=True)
                (3): Tanh()
                (4): Linear(in_features=20, out_features=20, bias=True)
                )
              )
              (branch_net): FeedForward(
                (extra_features): Sequential()
                (model): Sequential(
                (0): Linear(in_features=2, out_features=20, bias=True)
                (1): Tanh()
                (2): Linear(in_features=20, out_features=20, bias=True)
                (3): Tanh()
                (4): Linear(in_features=20, out_features=20, bias=True)
                )
              )
            )
        """
        super().__init__()

        self.output_variables = output_variables
        self.output_dimension = len(output_variables)

        trunk_out_dim = trunk_net.layers[-1].out_features
        branch_out_dim = branch_net.layers[-1].out_features

        if trunk_out_dim != branch_out_dim:
            raise ValueError('Branch and trunk networks have not the same '
                             'output dimension.')

        self.trunk_net = trunk_net
        self.branch_net = branch_net

        self.reduction = nn.Linear(trunk_out_dim, self.output_dimension)

    @property
    def input_variables(self):
        """The input variables of the model"""
        return self.trunk_net.input_variables + self.branch_net.input_variables

    def forward(self, x):
        """
        Defines the computation performed at every call.

        :param LabelTensor x: the input tensor.
        :return: the output computed by the model.
        :rtype: LabelTensor
        """

        branch_output = self.branch_net(
           x.extract(self.branch_net.input_variables))

        trunk_output = self.trunk_net(
           x.extract(self.trunk_net.input_variables))

        output_ = self.reduction(trunk_output * branch_output)

        output_ = output_.as_subclass(LabelTensor)
        output_.labels = self.output_variables

        return output_
