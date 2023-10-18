Tutorial 6: How to Use Geometries in PINA
=========================================

Built-in Geometries
-------------------

In this tutorial we will show how to use geometries in PINA.
Specifically, the tutorial will include how to create geometries and how
to visualize them. The topics covered are:

-  Creating CartesianDomains and EllipsoidDomains
-  Getting the Union and Difference of Geometries
-  Sampling points in the domain (and visualize them)

We import the relevant modules.

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pina.geometry import EllipsoidDomain, Difference, CartesianDomain, Union, SimplexDomain
    from pina.label_tensor import LabelTensor
    
    def plot_scatter(ax, pts, title):
        ax.title.set_text(title)
        ax.scatter(pts.extract('x'), pts.extract('y'), color='blue', alpha=0.5)

We will create one cartesian and two ellipsoids. For the sake of
simplicity, we show here the 2-dimensional, but it's trivial the
extension to 3D (and higher) cases. The geometries allows also the
generation of samples belonging to the boundary. So, we will create one
ellipsoid with the border and one without.

.. code:: ipython3

    cartesian = CartesianDomain({'x': [0, 2], 'y': [0, 2]})
    ellipsoid_no_border = EllipsoidDomain({'x': [1, 3], 'y': [1, 3]})
    ellipsoid_border = EllipsoidDomain({'x': [2, 4], 'y': [2, 4]}, sample_surface=True)

The ``{'x': [0, 2], 'y': [0, 2]}`` are the bounds of the
``CartesianDomain`` being created.

To visualize these shapes, we need to sample points on them. We will use
the ``sample`` method of the ``CartesianDomain`` and ``EllipsoidDomain``
classes. This method takes a ``n`` argument which is the number of
points to sample. It also takes different modes to sample such as
random.

.. code:: ipython3

    cartesian_samples = cartesian.sample(n=1000, mode='random')
    ellipsoid_no_border_samples = ellipsoid_no_border.sample(n=1000, mode='random')
    ellipsoid_border_samples = ellipsoid_border.sample(n=1000, mode='random')

We can see the samples of each of the geometries to see what we are
working with.

.. code:: ipython3

    print(f"Cartesian Samples: {cartesian_samples}")
    print(f"Ellipsoid No Border Samples: {ellipsoid_no_border_samples}")
    print(f"Ellipsoid Border Samples: {ellipsoid_border_samples}")


.. parsed-literal::

    Cartesian Samples: labels(['x', 'y'])
    LabelTensor([[[0.2300, 1.6698]],
                 [[1.7785, 0.4063]],
                 [[1.5143, 1.8979]],
                 ...,
                 [[0.0905, 1.4660]],
                 [[0.8176, 1.7357]],
                 [[0.0475, 0.0170]]])
    Ellipsoid No Border Samples: labels(['x', 'y'])
    LabelTensor([[[1.9341, 2.0182]],
                 [[1.5503, 1.8426]],
                 [[2.0392, 1.7597]],
                 ...,
                 [[1.8976, 2.2859]],
                 [[1.8015, 2.0012]],
                 [[2.2713, 2.2355]]])
    Ellipsoid Border Samples: labels(['x', 'y'])
    LabelTensor([[[3.3413, 3.9400]],
                 [[3.9573, 2.7108]],
                 [[3.8341, 2.4484]],
                 ...,
                 [[2.7251, 2.0385]],
                 [[3.8654, 2.4990]],
                 [[3.2292, 3.9734]]])


Notice how these are all ``LabelTensor`` objects. You can read more
about these in the
`documentation <https://mathlab.github.io/PINA/_rst/label_tensor.html>`__.
At a very high level, they are tensors where each element in a tensor
has a label that we can access by doing ``<tensor_name>.labels``. We can
also access the values of the tensor by doing
``<tensor_name>.extract(['x'])``.

We are now ready to visualize the samples using matplotlib.

.. code:: ipython3

    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    pts_list = [cartesian_samples, ellipsoid_no_border_samples, ellipsoid_border_samples]
    title_list = ['Cartesian Domain', 'Ellipsoid Domain', 'Ellipsoid Border Domain']
    for ax, pts, title in zip(axs, pts_list, title_list):
        plot_scatter(ax, pts, title)



.. image:: output_11_0.png


We have now created, sampled, and visualized our first geometries! We
can see that the ``EllipsoidDomain`` with the border has a border around
it. We can also see that the ``EllipsoidDomain`` without the border is
just the ellipse. We can also see that the ``CartesianDomain`` is just a
square.

Simplex Domain
~~~~~~~~~~~~~~

Among the built-in shapes, we quickly show here the usage of
``SimplexDomain``, which can be used for polygonal domains!

.. code:: ipython3

    import torch
    spatial_domain = SimplexDomain(
                        [
                            LabelTensor(torch.tensor([[0, 0]]), labels=["x", "y"]),
                            LabelTensor(torch.tensor([[1, 1]]), labels=["x", "y"]),
                            LabelTensor(torch.tensor([[0, 2]]), labels=["x", "y"]),
                        ]
                    )
    
    spatial_domain2 = SimplexDomain(
                        [
                            LabelTensor(torch.tensor([[ 0., -2.]]), labels=["x", "y"]),
                            LabelTensor(torch.tensor([[-.5, -.5]]), labels=["x", "y"]),
                            LabelTensor(torch.tensor([[-2.,  0.]]), labels=["x", "y"]),
                        ]
                    )
    
    pts = spatial_domain2.sample(100)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    for domain, ax in zip([spatial_domain, spatial_domain2], axs):
        pts = domain.sample(1000)
        plot_scatter(ax, pts, 'Simplex Domain')



.. image:: output_14_0.png


Boolean Operations
------------------

To create complex shapes we can use the boolean operations, for example
to merge two default geometries. We need to simply use the ``Union``
class: it takes a list of geometries and returns the union of them.

Let's create three unions. Firstly, it will be a union of ``cartesian``
and ``ellipsoid_no_border``. Next, it will be a union of
``ellipse_no_border`` and ``ellipse_border``. Lastly, it will be a union
of all three geometries.

.. code:: ipython3

    cart_ellipse_nb_union = Union([cartesian, ellipsoid_no_border])
    cart_ellipse_b_union = Union([cartesian, ellipsoid_border])
    three_domain_union = Union([cartesian, ellipsoid_no_border, ellipsoid_border])

We can of course sample points over the new geometries, by using the
``sample`` method as before. We highlihgt that the available sample
strategy here is only *random*.

.. code:: ipython3

    c_e_nb_u_points = cart_ellipse_nb_union.sample(n=2000, mode='random')
    c_e_b_u_points = cart_ellipse_b_union.sample(n=2000, mode='random')
    three_domain_union_points = three_domain_union.sample(n=3000, mode='random')

We can plot the samples of each of the unions to see what we are working
with.

.. code:: ipython3

    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    pts_list = [c_e_nb_u_points, c_e_b_u_points, three_domain_union_points]
    title_list = ['Cartesian with Ellipsoid No Border Union', 'Cartesian with Ellipsoid Border Union', 'Three Domain Union']
    for ax, pts, title in zip(axs, pts_list, title_list):
        plot_scatter(ax, pts, title)



.. image:: output_21_0.png


Now, we will find the differences of the geometries. We will find the
difference of ``cartesian`` and ``ellipsoid_no_border``.

.. code:: ipython3

    cart_ellipse_nb_difference = Difference([cartesian, ellipsoid_no_border])
    c_e_nb_d_points = cart_ellipse_nb_difference.sample(n=2000, mode='random')
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_scatter(ax, c_e_nb_d_points, 'Difference')



.. image:: output_23_0.png


Create Custom Location
----------------------

We will take a look on how to create our own geometry. The one we will
try to make is a heart defined by the function

.. math:: (x^2+y^2-1)^3-x^2y^3 \le 0

Let's start by importing what we will need to create our own geometry
based on this equation.

.. code:: ipython3

    import torch
    from pina import Location
    from pina import LabelTensor
    import random

Next, we will create the ``Heart(Location)`` class and initialize it.

.. code:: ipython3

    class Heart(Location):
        """Implementation of the Heart Domain."""
    
        def __init__(self, sample_border=False):
            super().__init__()
            

Because the ``Location`` class we are inherting from requires both a
sample method and ``is_inside`` method, we will create them and just add
in "pass" for the moment.

.. code:: ipython3

    class Heart(Location):
        """Implementation of the Heart Domain."""
    
        def __init__(self, sample_border=False):
            super().__init__()
    
        def is_inside(self):
            pass
    
        def sample(self):
            pass

Now we have the skeleton for our ``Heart`` class. The ``is_inside``
method is where most of the work is done so let's fill it out.

.. code:: ipython3

    
    class Heart(Location):
        """Implementation of the Heart Domain."""
    
        def __init__(self, sample_border=False):
            super().__init__()
    
        def is_inside(self):
            pass
    
        def sample(self, n, mode='random', variables='all'):
            sampled_points = []
    
            while len(sampled_points) < n:
                x = torch.rand(1)*3.-1.5
                y = torch.rand(1)*3.-1.5
                if ((x**2 + y**2 - 1)**3 - (x**2)*(y**3)) <= 0:
                    sampled_points.append([x.item(), y.item()])
    
            return LabelTensor(torch.tensor(sampled_points), labels=['x','y'])

To create the Heart geometry we simply run:

.. code:: ipython3

    heart = Heart()

To sample from the Heart geometry we simply run:

.. code:: ipython3

    pts_heart = heart.sample(1500)
    
    fig, ax = plt.subplots()
    plot_scatter(ax, pts_heart, 'Heart Domain')



.. image:: output_37_0.png

