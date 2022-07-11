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
    def __init__(self, branch_net, trunk_net, output_variables, inner_size=10,
                 features=None, features_net=None):
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

        # if features:
            # if len(features) != features_net.layers[0].in_features:
            #     raise ValueError('Incompatible features')
            # if trunk_out_dim != features_net.layers[-1].out_features:
            #     raise ValueError('Incompatible features')

            # self.features = features
            # self.features_net = nn.Sequential(
            #     nn.Linear(len(features), 10), nn.Softplus(),
            #     # nn.Linear(10, 10), nn.Softplus(),
            #     nn.Linear(10, trunk_out_dim)
            # )
            # self.features_net = nn.Sequential(
            #     nn.Linear(len(features), trunk_out_dim)
            # )

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
        # print(x.shape)
        #input_feature = []
        #for feature in self.features:
        #    #print(feature)
        #    input_feature.append(feature(x))
        #input_feature = torch.cat(input_feature, dim=1)

        branch_output = self.branch_net(
           x.extract(self.branch_net.input_variables))
        # print(branch_output.shape)
        trunk_output = self.trunk_net(
           x.extract(self.trunk_net.input_variables))
        # print(trunk_output.shape)
        #feat_output = self.features_net(input_feature)
        # print(feat_output.shape)
        # inner_input = torch.cat([
        #     branch_output * trunk_output,
        #     branch_output,
        #     trunk_output,
        #     feat_output], dim=1)
        # print(inner_input.shape)

        # output_ = self.reduction(inner_input)
        # print(output_.shape)
        print(branch_output.shape)
        print(trunk_output.shape)
        output_ = self.reduction(trunk_output * branch_output)
        output_ = LabelTensor(output_, self.output_variables)
        # local_size = int(trunk_output.shape[1]/self.output_dimension)
        # for i, var in enumerate(self.output_variables):
        #     start = i*local_size
        #     stop = (i+1)*local_size
        #     local_output = LabelTensor(torch.sum(branch_output[:, start:stop] * trunk_output[:, start:stop], dim=1).reshape(-1, 1), var)
        #     if i==0:
        #         output_ = local_output
        #     else:
        #         output_ = output_.append(local_output)
        return output_
