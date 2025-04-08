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
    "DeepEnsembleSolverInterface",
    "DeepEnsembleSupervisedSolver",
    "DeepEnsemblePINN",
    "GAROM",
]

from .solver import SolverInterface, SingleSolverInterface, MultiSolverInterface
from .physics_informed_solver import *
from .supervised_solver import *
from .ensemble_solver import *
from .garom import GAROM
