"""Module for the solver classes."""

__all__ = [
    "SolverInterface",
    "SingleSolverInterface",
    "MultiSolverInterface",
    "PINNInterface",
    "PINN",
    "GradientPINN",
    "CausalPINN",
    "CompetitivePINN",
    "SelfAdaptivePINN",
    "RBAPINN",
    "SupervisedSolver",
    "ReducedOrderModelSolver",
    "GAROM",
]

from .solver import SolverInterface, SingleSolverInterface, MultiSolverInterface
from .physics_informed_solver import *
from .supervised import SupervisedSolver
from .reduced_order_model import ReducedOrderModelSolver
from .garom import GAROM
