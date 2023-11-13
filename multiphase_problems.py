import torch

from pina.problem import TimeDependentProblem, SpatialProblem
from pina.operators import grad
from pina import Condition, LabelTensor
from pina.geometry import CartesianDomain
from pina.equation.equation import Equation
from pina.equation.equation_factory import FixedValue
from pina.equation import Equation

class AdvectionEquation(Equation):

    def __init__(self, velocity_field, components=None):

        self.components = components

        def advection(input_, output_):
            if self.components is None:
                components = input_.labels
            else:
                components = self.components 
            grad_ = grad(output_, input_)
            dt = grad_.extract(['dphidt'])
            dphi = grad_.extract(['dphidx', 'dphidy'])
            u0 = velocity_field(input_.extract(components))
            return dt + torch.einsum("ij,ik->i", u0, dphi).reshape(-1, 1)
        super().__init__(advection)

class ZalesakDisck(TimeDependentProblem, SpatialProblem):

    
    # assign output/ spatial and temporal variables
    output_variables = ['phi']
    spatial_domain = CartesianDomain({'x': [0, 1], 'y': [-0.5, 0.5]})
    temporal_domain = CartesianDomain({'t': [0, 4]})


    @staticmethod
    def phi_initial(pts):
        eps = 3e-4
        center = LabelTensor(torch.tensor([[0.5, 0.0]]), labels=['x', 'y'])
        x = pts.extract(['x'])
        y = pts.extract(['y'])
        radius = torch.sqrt( (x-center.extract(['x']))**2 + (y-center.extract(['y']))**2 )
        h_r = 0.5 * ( 1. + torch.tanh( (radius-0.15) / (2 * eps) ) )
        h_y = 0.5 * ( 1. + torch.tanh( (y-0.1) / (2 * eps) ) )
        h_x = 0.5 * ( 1. + torch.tanh( (torch.abs(x-0.5) - 0.025) / (2 * eps) ) )
        return (1. - h_r) * (1. - (1. - h_y) * (1. - h_x) )
    
    # initial velocity
    def velocity_field(input_):
        x = input_.extract(['x'])
        y = input_.extract(['y'])
        u1 = (x - 0.5)
        u2 = - y
        return torch.hstack((u1, u2))

    # define initial condition
    def initial_condition(input_, output_):
        phi_expected = ZalesakDisck.phi_initial(input_)
        return output_.extract(['phi']) - phi_expected

    # problem condition statement
    conditions = {
        'gamma1': Condition(
            location=CartesianDomain({'x': [0, 1], 'y': -0.5, 't' : [0, 4]}),
            equation=FixedValue(0.0)),
        'gamma2': Condition(
            location=CartesianDomain({'x': [0, 1], 'y': 0.5, 't' : [0, 4]}),
            equation=FixedValue(0.0)),
        'gamma3': Condition(
            location=CartesianDomain({'x': 0, 'y': [-0.5, 0.5], 't' : [0, 4]}),
            equation=FixedValue(0.0)),
        'gamma4': Condition(
            location=CartesianDomain({'x': 1, 'y': [-0.5, 0.5], 't' : [0, 4]}),
            equation=FixedValue(0.0)),
        't0': Condition(location=CartesianDomain({'x': [0, 1], 'y': [-0.5, 0.5], 't': 0}), equation=Equation(initial_condition)),
        'D': Condition(location=CartesianDomain({'x': [0, 1], 'y': [-0.5, 0.5], 't': [0.001, 4]}), equation=AdvectionEquation(velocity_field)),
    }








class RiderKotheVortex(TimeDependentProblem, SpatialProblem):


    # assign output/ spatial and temporal variables
    output_variables = ['phi']
    spatial_domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})
    temporal_domain = CartesianDomain({'t': [0, 4]})

    @staticmethod
    def phi_initial(input_):
        eps = 3e-3
        x = input_.extract(['x']) - 0.5
        y = input_.extract(['y']) - 0.75
        norm_ = torch.sqrt(x**2 + y**2)
        phi_expected = 0.5*(1. + torch.tanh((norm_ - 0.15) / (2.*eps)))
        return phi_expected

    # initial velocity
    def velocity_field(input_):
        x = input_.extract(['x'])
        y = input_.extract(['y'])
        t = input_.extract(['t'])
        u1 = torch.sin(torch.pi*x).pow(2)*torch.sin(2.*torch.pi*y)*torch.cos(torch.pi*t/4)
        u2 = -torch.sin(torch.pi*y).pow(2)*torch.sin(2.*torch.pi*x)*torch.cos(torch.pi*t/4)
        return torch.hstack((u1, u2))

    # define initial condition
    def initial_condition(input_, output_):
        phi_expected = RiderKotheVortex.phi_initial(input_)
        return output_.extract(['phi']) - phi_expected
    
    def mass_conservation(input_, output_):
        phi_expected = RiderKotheVortex.phi_initial(input_)
        tot_mass = phi_expected.sum(-1)
        out_mass = output_.sum(-1)
        return (tot_mass - out_mass).reshape(-1, 1)
    
    def advection(input_, output_):
        grad_ = grad(output_, input_)
        dt = grad_.extract(['dphidt'])
        dphi = grad_.extract(['dphidx', 'dphidy'])
        u0 = RiderKotheVortex.velocity_field(input_)
        return dt + torch.einsum("ij,ik->i", u0, dphi).reshape(-1, 1)

    # problem condition statement
    conditions = {
        'gamma1': Condition(
            location=CartesianDomain({'x': [0, 1], 'y': 0, 't' : [0, 4]}),
            equation=FixedValue(1.0)),
        'gamma2': Condition(
            location=CartesianDomain({'x': [0, 1], 'y': 1, 't' : [0, 4]}),
            equation=FixedValue(1.0)),
        'gamma3': Condition(
            location=CartesianDomain({'x': 0, 'y': [0, 1], 't' : [0, 4]}),
            equation=FixedValue(1.0)),
        'gamma4': Condition(
            location=CartesianDomain({'x': 1, 'y': [0, 1], 't' : [0, 4]}),
            equation=FixedValue(1.0)),
        't0': Condition(location=CartesianDomain({'x': [0, 1], 'y': [0, 1], 't': 0}), equation=Equation(initial_condition)),
        'D': Condition(location=CartesianDomain({'x': [0, 1], 'y': [0, 1], 't': [0, 4]}), equation= AdvectionEquation(velocity_field)),
    }