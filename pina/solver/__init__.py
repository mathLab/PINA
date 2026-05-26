"""Module for all solvers in PINA."""

__all__ = [
    "SolverInterface",
    "BaseSolver",
    "SingleModelSolver",
    "MultiModelSolver",
    "EnsembleSolver",
    "SupervisedSingleModelSolver",
    "PhysicsInformedSingleModelSolver",
    "SupervisedEnsembleSolver",
    "PhysicsInformedEnsembleSolver",
    "AutoregressiveSingleModelSolver",
    "AutoregressiveEnsembleSolver",
]


from pina._src.solver.solver_interface import SolverInterface
from pina._src.solver.base_solver import BaseSolver
from pina._src.solver.single_model_solver import SingleModelSolver
from pina._src.solver.multi_model_solver import MultiModelSolver
from pina._src.solver.ensemble_solver import EnsembleSolver
from pina._src.solver.supervised_single_model_solver import (
    SupervisedSingleModelSolver,
)
from pina._src.solver.physics_informed_single_model_solver import (
    PhysicsInformedSingleModelSolver,
)
from pina._src.solver.supervised_ensemble_solver import SupervisedEnsembleSolver
from pina._src.solver.physics_informed_ensemble_solver import (
    PhysicsInformedEnsembleSolver,
)
from pina._src.solver.autoregressive_single_model_solver import (
    AutoregressiveSingleModelSolver,
)
from pina._src.solver.autoregressive_ensemble_solver import (
    AutoregressiveEnsembleSolver,
)
