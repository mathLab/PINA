from pina.domain import CartesianDomain
from pina import Condition, Plotter
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from pina.problem import SpatialProblem
from pina.equation import FixedValue

"""

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
TODO : Fix the tests once the Plotter class is updated
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

class FooProblem1D(SpatialProblem):

    # assign output/ spatial and temporal variables
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x' : [-1, 1]})

    # problem condition statement
    conditions = {
        'D': Condition(location=CartesianDomain({'x': [-1, 1]}), equation=FixedValue(0.)),
    }

class FooProblem2D(SpatialProblem):

    # assign output/ spatial and temporal variables
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x' : [-1, 1], 'y': [-1, 1]})

    # problem condition statement
    conditions = {
        'D': Condition(location=CartesianDomain({'x' : [-1, 1], 'y': [-1, 1]}), equation=FixedValue(0.)),
    }

class FooProblem3D(SpatialProblem):

    # assign output/ spatial and temporal variables
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x' : [-1, 1], 'y': [-1, 1], 'z':[-1,1]})

    # problem condition statement
    conditions = {
        'D': Condition(location=CartesianDomain({'x' : [-1, 1], 'y': [-1, 1], 'z':[-1,1]}), equation=FixedValue(0.)),
    }



def test_constructor():
    Plotter()

def test_plot_samples_1d():
    problem = FooProblem1D()
    problem.discretise_domain(n=10, mode='grid', variables = 'x', locations=['D'])
    pl = Plotter()
    pl.plot_samples(problem=problem, filename='fig.png')
    import os
    os.remove('fig.png')

def test_plot_samples_2d():
    problem = FooProblem2D()
    problem.discretise_domain(n=10, mode='grid', variables = ['x', 'y'], locations=['D'])
    pl = Plotter()
    pl.plot_samples(problem=problem, filename='fig.png')
    import os
    os.remove('fig.png')

def test_plot_samples_3d():
    problem = FooProblem3D()
    problem.discretise_domain(n=10, mode='grid', variables = ['x', 'y', 'z'], locations=['D'])
    pl = Plotter()
    pl.plot_samples(problem=problem, filename='fig.png')
    import os
    os.remove('fig.png')
"""