Tutorial 2: resolution of Poisson problem and usage of extra-features
=====================================================================

The problem definition
~~~~~~~~~~~~~~~~~~~~~~

This tutorial presents how to solve with Physics-Informed Neural
Networks a 2D Poisson problem with Dirichlet boundary conditions.

The problem is written as: :raw-latex:`\begin{equation}
\begin{cases}
\Delta u = \sin{(\pi x)} \sin{(\pi y)} \text{ in } D, \\
u = 0 \text{ on } \Gamma_1 \cup \Gamma_2 \cup \Gamma_3 \cup \Gamma_4,
\end{cases}
\end{equation}` where :math:`D` is a square domain :math:`[0,1]^2`, and
:math:`\Gamma_i`, with :math:`i=1,...,4`, are the boundaries of the
square.

First of all, some useful imports.

.. code:: ipython3

    import torch
    from torch.nn import Softplus
    
    from pina.problem import SpatialProblem
    from pina.operators import nabla
    from pina.model import FeedForward
    from pina import Condition, Span, PINN, LabelTensor, Plotter

Now, the Poisson problem is written in PINA code as a class. The
equations are written as *conditions* that should be satisfied in the
corresponding domains. *truth_solution* is the exact solution which will
be compared with the predicted one.

.. code:: ipython3

    class Poisson(SpatialProblem):
        output_variables = ['u']
        spatial_domain = Span({'x': [0, 1], 'y': [0, 1]})
    
        def laplace_equation(input_, output_):
            force_term = (torch.sin(input_.extract(['x'])*torch.pi) *
                          torch.sin(input_.extract(['y'])*torch.pi))
            nabla_u = nabla(output_, input_, components=['u'], d=['x', 'y'])
            return nabla_u - force_term
    
        def nil_dirichlet(input_, output_):
            value = 0.0
            return output_.extract(['u']) - value
    
        conditions = {
            'gamma1': Condition(location=Span({'x': [0, 1], 'y':  1}), function=nil_dirichlet),
            'gamma2': Condition(location=Span({'x': [0, 1], 'y': 0}), function=nil_dirichlet),
            'gamma3': Condition(location=Span({'x':  1, 'y': [0, 1]}), function=nil_dirichlet),
            'gamma4': Condition(location=Span({'x': 0, 'y': [0, 1]}), function=nil_dirichlet),
            'D': Condition(location=Span({'x': [0, 1], 'y': [0, 1]}), function=laplace_equation),
        }
    
        def poisson_sol(self, pts):
            return -(
                torch.sin(pts.extract(['x'])*torch.pi)*
                torch.sin(pts.extract(['y'])*torch.pi)
            )/(2*torch.pi**2)
        
        truth_solution = poisson_sol

The problem solution
~~~~~~~~~~~~~~~~~~~~

After the problem, the feed-forward neural network is defined, through
the class ``FeedForward``. This neural network takes as input the
coordinates (in this case :math:`x` and :math:`y`) and provides the
unkwown field of the Poisson problem. The residual of the equations are
evaluated at several sampling points (which the user can manipulate
using the method ``span_pts``) and the loss minimized by the neural
network is the sum of the residuals.

In this tutorial, the neural network is composed by two hidden layers of
10 neurons each, and it is trained for 1000 epochs with a learning rate
of 0.006. These parameters can be modified as desired. The output of the
cell below is the final loss of the training phase of the PINN. We
highlight that the generation of the sampling points and the train is
here encapsulated within the function ``generate_samples_and_train``,
but only for saving some lines of code in the next cells; that function
is not mandatory in the **PINA** framework.

.. code:: ipython3

    def generate_samples_and_train(model, problem):
        pinn = PINN(problem, model, lr=0.006, regularizer=1e-8)
        pinn.span_pts(20, 'grid', locations=['D'])
        pinn.span_pts(20, 'grid', locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])
        pinn.train(1000, 100)
        return pinn
    
    problem = Poisson()
    model = FeedForward(
        layers=[10, 10],
        func=Softplus,
        output_variables=problem.output_variables,
        input_variables=problem.input_variables
    )
    
    pinn = generate_samples_and_train(model, problem)


.. parsed-literal::

                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00000] 4.879922e-01 1.557781e-01 7.685463e-02 2.743466e-02 2.047883e-02 2.074460e-01 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00001] 2.610107e-01 1.067532e-03 8.390929e-03 2.391219e-02 1.467707e-02 2.129630e-01 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00100] 8.640952e-02 1.038323e-04 9.709063e-05 6.688796e-05 6.651071e-05 8.607519e-02 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00200] 2.996790e-02 4.977722e-04 6.639907e-04 5.634258e-04 7.204801e-04 2.752223e-02 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00300] 2.896983e-03 1.864277e-04 2.020803e-05 2.418693e-04 3.052877e-05 2.417949e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00400] 1.865673e-03 1.250375e-04 2.438288e-05 1.595948e-04 6.709602e-06 1.549948e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00500] 2.874877e-03 2.077810e-04 1.149128e-04 1.273361e-04 3.024802e-06 2.421822e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00600] 1.310072e-03 1.081258e-04 3.365631e-05 1.059794e-04 3.468987e-06 1.058841e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00700] 2.694587e-03 1.267468e-04 6.266955e-05 9.891923e-05 8.897325e-06 2.397354e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00800] 5.028690e-03 1.435707e-04 5.986574e-06 9.517078e-05 4.583780e-05 4.738124e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00900] 9.997603e-04 9.684711e-05 9.155992e-06 8.875966e-05 1.261154e-05 7.923861e-04 
    [epoch 01000] 2.362966e-02 1.157872e-04 7.812096e-06 8.004917e-05 9.947084e-05 2.332654e-02 


The neural network of course can be saved in a file. In such a way, we
can store it after the train, and load it just to infer the field. Here
we don’t store the model, but for demonstrative purposes we put in the
next cell the commented line of code.

.. code:: ipython3

    # pinn.save_state('pina.poisson')

Now the *Plotter* class is used to plot the results. The solution
predicted by the neural network is plotted on the left, the exact one is
represented at the center and on the right the error between the exact
and the predicted solutions is showed.

.. code:: ipython3

    plotter = Plotter()
    plotter.plot(pinn)



.. image:: tutorial_files/tutorial_13_0.png


The problem solution with extra-features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, the same problem is solved in a different way. A new neural network
is now defined, with an additional input variable, named extra-feature,
which coincides with the forcing term in the Laplace equation. The set
of input variables to the neural network is:

:raw-latex:`\begin{equation}
[x, y, k(x, y)], \text{ with } k(x, y)=\sin{(\pi x)}\sin{(\pi y)},
\end{equation}`

where :math:`x` and :math:`y` are the spatial coordinates and
:math:`k(x, y)` is the added feature.

This feature is initialized in the class ``SinSin``, which needs to be
inherited by the ``torch.nn.Module`` class and to have the ``forward``
method. After declaring such feature, we can just incorporate in the
``FeedForward`` class thanks to the ``extra_features`` argument. **NB**:
``extra_features`` always needs a ``list`` as input, you you have one
feature just encapsulated it in a class, as in the next cell.

Finally, we perform the same training as before: the problem is
``Poisson``, the network is composed by the same number of neurons and
optimizer parameters are equal to previous test, the only change is the
new extra feature.

.. code:: ipython3

    class SinSin(torch.nn.Module):
        """Feature: sin(x)*sin(y)"""
        def __init__(self):
            super().__init__()
    
        def forward(self, x):
            t = (torch.sin(x.extract(['x'])*torch.pi) *
                 torch.sin(x.extract(['y'])*torch.pi))
            return LabelTensor(t, ['sin(x)sin(y)'])
    
    model_feat = FeedForward(
            layers=[10, 10],
            output_variables=problem.output_variables,
            input_variables=problem.input_variables,
            func=Softplus,
            extra_features=[SinSin()]
        )
    
    pinn_feat = generate_samples_and_train(model_feat, problem)


.. parsed-literal::

                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00000] 1.309440e-01 2.335824e-02 3.823499e-03 1.878588e-05 2.002613e-03 1.017409e-01 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00001] 5.053994e-02 6.420787e-03 6.924602e-03 4.746807e-03 1.751946e-03 3.069580e-02 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00100] 7.484706e-06 1.889349e-07 4.289622e-07 3.610726e-07 3.611258e-07 6.144610e-06 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00200] 6.941436e-06 4.738185e-07 4.590637e-07 5.098815e-07 5.365398e-07 4.962133e-06 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00300] 6.147081e-06 6.213511e-07 5.576677e-07 6.256337e-07 6.572442e-07 3.685184e-06 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00400] 6.056770e-06 7.646217e-07 6.377599e-07 7.242416e-07 7.616553e-07 3.168491e-06 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00500] 6.751128e-06 8.011474e-07 6.283512e-07 7.652199e-07 7.226305e-07 3.833779e-06 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00600] 2.839740e-05 5.422368e-06 4.058312e-06 4.664194e-06 4.984503e-06 9.268020e-06 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00700] 1.221099e-05 3.654685e-06 3.195583e-07 2.717753e-06 2.381476e-06 3.137519e-06 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00800] 5.423951e-06 6.111856e-07 4.348901e-07 5.353588e-07 5.398895e-07 3.302627e-06 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00900] 6.777007e-06 3.749606e-07 1.421852e-06 4.068826e-08 1.292241e-06 3.647265e-06 
    [epoch 01000] 6.803403e-05 2.302543e-07 3.886034e-05 4.901193e-06 2.005441e-05 3.987827e-06 


The predicted and exact solutions and the error between them are
represented below. We can easily note that now our network, having
almost the same condition as before, is able to reach an additional
order of magnitude in accuracy.

.. code:: ipython3

    plotter.plot(pinn_feat)



.. image:: tutorial_files/tutorial_18_0.png


The problem solution with learnable extra-features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can still do better!

Another way to exploit the extra features is the addition of learnable
parameter inside them. In this way, the added parameters are learned
during the training phase of the neural network. In this case, we use:

:raw-latex:`\begin{equation}
k(x, \mathbf{y}) = \beta \sin{(\alpha x)} \sin{(\alpha y)},
\end{equation}`

where :math:`\alpha` and :math:`\beta` are the abovementioned
parameters. Their implementation is quite trivial: by using the class
``torch.nn.Parameter`` we cam define all the learnable parameters we
need, and they are managed by ``autograd`` module!

.. code:: ipython3

    class SinSinAB(torch.nn.Module):
        """ """
        def __init__(self):
            super().__init__()
            self.alpha = torch.nn.Parameter(torch.tensor([1.0]))
            self.beta = torch.nn.Parameter(torch.tensor([1.0]))
    
    
        def forward(self, x):
            t =  (
                self.beta*torch.sin(self.alpha*x.extract(['x'])*torch.pi)*
                          torch.sin(self.alpha*x.extract(['y'])*torch.pi)
            )
            return LabelTensor(t, ['b*sin(a*x)sin(a*y)'])
    
    
    model_learn = FeedForward(
        layers=[10, 10],
        output_variables=problem.output_variables,
        input_variables=problem.input_variables,
        extra_features=[SinSinAB()]
    )
    
    pinn_learn = generate_samples_and_train(model_learn, problem)


.. parsed-literal::

                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00000] 7.147130e-02 1.942330e-03 7.350697e-03 2.868338e-03 1.184232e-03 5.812570e-02 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00001] 2.814954e-01 7.300152e-03 5.510583e-04 2.262258e-03 7.287678e-04 2.706531e-01 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00100] 1.961870e-04 3.066778e-06 5.342949e-07 2.670689e-06 9.807675e-07 1.889345e-04 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00200] 1.208203e-04 3.096610e-06 1.253595e-06 2.603416e-06 1.962141e-06 1.119046e-04 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00300] 3.992990e-05 3.451424e-06 6.415143e-07 1.576505e-06 1.244609e-06 3.301585e-05 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00400] 3.466437e-04 1.722332e-06 1.461791e-05 3.052185e-06 8.755493e-06 3.184958e-04 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00500] 5.242374e-03 3.230991e-05 1.387528e-05 5.379211e-06 3.145076e-06 5.187664e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00600] 1.027368e-03 1.448758e-06 2.165510e-05 5.197179e-05 3.823021e-05 9.140619e-04 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00700] 1.141694e-03 6.998039e-06 2.446730e-05 3.083524e-05 1.376935e-05 1.065624e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00800] 3.619534e-04 3.120772e-06 1.223103e-05 2.211869e-05 9.567964e-06 3.149150e-04 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00900] 3.287693e-04 2.432459e-06 7.569996e-06 1.101516e-05 4.546776e-06 3.032049e-04 
    [epoch 01000] 5.432598e-04 8.919213e-06 1.991732e-05 2.632461e-05 7.365395e-06 4.807333e-04 


Umh, the final loss is not appreciabily better than previous model (with
static extra features), despite the usage of learnable parameters. This
is mainly due to the over-parametrization of the network: there are many
parameter to optimize during the training, and the model in unable to
understand automatically that only the parameters of the extra feature
(and not the weights/bias of the FFN) should be tuned in order to fit
our problem. A longer training can be helpful, but in this case the
faster way to reach machine precision for solving the Poisson problem is
removing all the hidden layers in the ``FeedForward``, keeping only the
:math:`\alpha` and :math:`\beta` parameters of the extra feature.

.. code:: ipython3

    model_learn = FeedForward(
        layers=[],
        output_variables=problem.output_variables,
        input_variables=problem.input_variables,
        extra_features=[SinSinAB()]
    )
    
    pinn_learn = generate_samples_and_train(model_learn, problem)


.. parsed-literal::

                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00000] 1.907039e+01 5.862396e-02 5.423664e-01 4.624593e-01 7.118504e-02 1.793576e+01 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00001] 1.698682e+01 3.348809e-02 4.943427e-01 3.972439e-01 6.141453e-02 1.600033e+01 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00100] 8.010766e-02 1.765875e-04 6.100491e-04 1.604862e-04 5.841496e-04 7.857639e-02 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00200] 5.057434e-02 6.479959e-05 6.590948e-05 6.376287e-05 5.975253e-05 5.032011e-02 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00300] 1.974927e-02 3.145394e-05 1.531348e-05 3.037518e-05 1.363940e-05 1.965849e-02 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00400] 1.763019e-03 3.408035e-06 8.902280e-07 3.228933e-06 7.512407e-07 1.754741e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00500] 2.604023e-05 5.248935e-08 1.091775e-08 4.940254e-08 9.077334e-09 2.591834e-05 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00600] 7.279636e-08 1.490485e-10 3.004504e-11 1.392443e-10 2.490262e-11 7.245312e-08 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00700] 2.307051e-11 5.051121e-14 1.083412e-14 4.412749e-14 8.684963e-15 2.295635e-11 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00800] 9.755044e-12 1.745244e-14 3.232219e-15 1.735542e-14 3.347362e-15 9.713657e-12 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di Dlaplace_equ 
    [epoch 00900] 5.909113e-12 1.112281e-14 2.037945e-15 1.107687e-14 2.124603e-15 5.882751e-12 
    [epoch 01000] 3.220371e-12 5.622761e-15 1.002551e-15 5.519723e-15 9.455284e-16 3.207280e-12 


In such a way, the model is able to reach a very high accuracy! Of
course, this is a toy problem for understanding the usage of extra
features: similar precision could be obtained if the extra features are
very similar to the true solution. The analyzed Poisson problem shows a
forcing term very close to the solution, resulting in a perfect problem
to address with such an approach.

We conclude here by showing the graphical comparison of the unknown
field and the loss trend for all the test cases presented here: the
standard PINN, PINN with extra features, and PINN with learnable extra
features.

.. code:: ipython3

    plotter.plot(pinn_learn)



.. image:: tutorial_files/tutorial_25_0.png


.. code:: ipython3

    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(16, 6))
    plotter.plot_loss(pinn, label='Standard')
    plotter.plot_loss(pinn_feat, label='Static Features')
    plotter.plot_loss(pinn_learn, label='Learnable Features')
    
    plt.grid()
    plt.legend()
    plt.show()



.. image:: tutorial_files/tutorial_26_0.png

