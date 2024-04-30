""" Module for ReducedOrderModelSolver """

import torch

from pina.solvers import SupervisedSolver

class ReducedOrderModelSolver(SupervisedSolver):
    r"""
    ReducedOrderModelSolver solver class. This class implements a
    Reduced Order Model solver, using user specified ``reduction_network`` and
    ``interpolation_network`` to solve a specific ``problem``.

    The  Reduced Order Model approach aims to find
    the solution :math:`\mathbf{u}:\Omega\rightarrow\mathbb{R}^m`
    of the differential problem:

    .. math::

        \begin{cases}
        \mathcal{A}[\mathbf{u}(\mu)](\mathbf{x})=0\quad,\mathbf{x}\in\Omega\\
        \mathcal{B}[\mathbf{u}(\mu)](\mathbf{x})=0\quad,
        \mathbf{x}\in\partial\Omega
        \end{cases}

    This is done by using two neural networks. The ``reduction_network``, which
    contains an encoder :math:`\mathcal{E}_{\rm{net}}`, a decoder
    :math:`\mathcal{D}_{\rm{net}}`; and an ``interpolation_network``
    :math:`\mathcal{I}_{\rm{net}}`. The input is assumed to be discretised in
    the spatial dimensions.

    The following loss function is minimized during training

    .. math::
        \mathcal{L}_{\rm{problem}} = \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(\mathcal{E}_{\rm{net}}[\mathbf{u}(\mu_i)] -
        \mathcal{I}_{\rm{net}}[\mu_i]) + 
        \mathcal{L}(
            \mathcal{D}_{\rm{net}}[\mathcal{E}_{\rm{net}}[\mathbf{u}(\mu_i)]] -
            \mathbf{u}(\mu_i))

    where :math:`\mathcal{L}` is a specific loss function, default Mean Square Error:

    .. math::
        \mathcal{L}(v) = \| v \|^2_2.


    .. seealso::

        **Original reference**: Hesthaven, Jan S., and Stefano Ubbiali.
        "Non-intrusive reduced order modeling of nonlinear problems
        using neural networks." Journal of Computational
        Physics 363 (2018): 55-78.
        DOI `10.1016/j.jcp.2018.02.037
        <https://doi.org/10.1016/j.jcp.2018.02.037>`_.
        
    .. note::
        The specified ``reduction_network`` must contain two methods,
        namely ``encode`` for input encoding and ``decode`` for decoding the
        former result. The ``interpolation_network`` network ``forward`` output
        represents the interpolation of the latent space obtain with
        ``reduction_network.encode``.

    .. note::
        This solver uses the end-to-end training strategy, i.e. the
        ``reduction_network`` and ``interpolation_network`` are trained
        simultaneously. For reference on this trainig strategy look at:
        Pichi, Federico, Beatriz Moya, and Jan S. Hesthaven. 
        "A graph convolutional autoencoder approach to model order reduction
        for parametrized PDEs." Journal of
        Computational Physics 501 (2024): 112762.
        DOI 
        `10.1016/j.jcp.2024.112762 <https://doi.org/10.1016/
        j.jcp.2024.112762>`_.

    .. warning::
        This solver works only for data-driven model. Hence in the ``problem``
        definition the codition must only contain ``input_points``
        (e.g. coefficient parameters, time parameters), and ``output_points``.

    .. warning::
        This solver does not currently support the possibility to pass
        ``extra_feature``.
    """

    def __init__(
        self,
        problem,
        reduction_network,
        interpolation_network,
        loss=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.001},
        scheduler=torch.optim.lr_scheduler.ConstantLR,
        scheduler_kwargs={"factor": 1, "total_iters": 0},
    ):
        """
        :param AbstractProblem problem: The formualation of the problem.
        :param torch.nn.Module reduction_network: The reduction network used
            for reducing the input space. It must contain two methods,
            namely ``encode`` for input encoding and ``decode`` for decoding the
            former result.
        :param torch.nn.Module interpolation_network: The interpolation network
            for interpolating the control parameters to latent space obtain by
            the ``reduction_network`` encoding.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default :class:`torch.nn.MSELoss`.
        :param torch.nn.Module extra_features: The additional input
            features to use as augmented input.
        :param torch.optim.Optimizer optimizer: The neural network optimizer to
            use; default is :class:`torch.optim.Adam`.
        :param dict optimizer_kwargs: Optimizer constructor keyword args.
        :param float lr: The learning rate; default is 0.001.
        :param torch.optim.LRScheduler scheduler: Learning
            rate scheduler.
        :param dict scheduler_kwargs: LR scheduler constructor keyword args.
        """
        model = torch.nn.ModuleDict({
            'reduction_network' : reduction_network,
            'interpolation_network' : interpolation_network})
        
        super().__init__(
            model=model,
            problem=problem,
            loss=loss,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs
        )

        # assert reduction object contains encode/ decode
        if not hasattr(self.neural_net['reduction_network'], 'encode'):
            raise SyntaxError('reduction_network must have encode method. '
                               'The encode method should return a lower '
                               'dimensional representation of the input.')
        if not hasattr(self.neural_net['reduction_network'], 'decode'):
            raise SyntaxError('reduction_network must have decode method. '
                               'The decode method should return a high '
                               'dimensional representation of the encoding.')

    def forward(self, x):
        """
        Forward pass implementation for the solver. It finds the encoder
        representation by calling ``interpolation_network.forward`` on the
        input, and maps this representation to output space by calling
        ``reduction_network.decode``.

        :param torch.Tensor x: Input tensor.
        :return: Solver solution.
        :rtype: torch.Tensor
        """
        reduction_network = self.neural_net['reduction_network']
        interpolation_network = self.neural_net['interpolation_network']
        return reduction_network.decode(interpolation_network(x))

    def loss_data(self, input_pts, output_pts):
        """
        The data loss for the ReducedOrderModelSolver solver.
        It computes the loss between
        the network output against the true solution. This function
        should not be override if not intentionally.

        :param LabelTensor input_tensor: The input to the neural networks.
        :param LabelTensor output_tensor: The true solution to compare the
            network solution.
        :return: The residual loss averaged on the input coordinates
        :rtype: torch.Tensor
        """
        # extract networks
        reduction_network = self.neural_net['reduction_network']
        interpolation_network = self.neural_net['interpolation_network']
        # encoded representations loss
        encode_repr_inter_net = interpolation_network(input_pts)
        encode_repr_reduction_network = reduction_network.encode(output_pts)
        loss_encode = self.loss(encode_repr_inter_net,
                                encode_repr_reduction_network)
        # reconstruction loss
        loss_reconstruction = self.loss(
            reduction_network.decode(encode_repr_reduction_network),
            output_pts)

        return loss_encode + loss_reconstruction

    @property
    def neural_net(self):
        """
        Neural network for training. It returns a :obj:`~torch.nn.ModuleDict`
        containing the ``reduction_network`` and ``interpolation_network``.
        """
        return self._neural_net.torchmodel
