"""
Unified solvers for Physics-Informed and Data-Driven modeling.

This module provides the high-level training orchestrators used to solve
differential equations and regression problems. It includes:
* **Physics-Informed Solvers**: Standard PINN, Gradient-enhanced (gPINN), Causal,
  and Self-Adaptive variants.
* **Supervised Solvers**: For purely data-driven tasks and Reduced Order Modeling.
* **Ensemble Solvers**: For uncertainty quantification via Deep Ensembles.
"""

__all__ = [
    "SolverInterface",
    "SingleSolverInterface",
    "MultiSolverInterface",
    "SingleModelSimpleSolver",
    "PINNInterface",
    "PINN",
    "GradientPINN",
    "CausalPINN",
    "CompetitivePINN",
    "SelfAdaptivePINN",
    "RBAPINN",
    "SupervisedSolverInterface",
    "SupervisedSolver",
    "ReducedOrderModelSolver",
    "DeepEnsembleSolverInterface",
    "DeepEnsembleSupervisedSolver",
    "DeepEnsemblePINN",
    "GAROM",
    "AutoregressiveSolver",
]

from pina._src.solver.solver import (
    SolverInterface,
    SingleSolverInterface,
    MultiSolverInterface,
)
from pina._src.solver.single_model_simple_solver import (
    SingleModelSimpleSolver,
)
from pina._src.solver.pinn import PINNInterface, PINN
from pina._src.solver.physics_informed_solver.gradient_pinn import GradientPINN
from pina._src.solver.physics_informed_solver.causal_pinn import CausalPINN
from pina._src.solver.physics_informed_solver.competitive_pinn import (
    CompetitivePINN,
)
from pina._src.solver.physics_informed_solver.self_adaptive_pinn import (
    SelfAdaptivePINN,
)
from pina._src.solver.physics_informed_solver.rba_pinn import RBAPINN
from pina._src.solver.supervised_solver.supervised_solver_interface import (
    SupervisedSolverInterface,
)

from pina._src.solver.supervised_solver.supervised_solver_interface import (
    SupervisedSolverInterface,
)
from pina._src.solver.supervised import SupervisedSolver
from pina._src.solver.supervised_solver.reduced_order_model import (
    ReducedOrderModelSolver,
)
from pina._src.solver.ensemble_solver.ensemble_solver_interface import (
    DeepEnsembleSolverInterface,
)
from pina._src.solver.ensemble_solver.ensemble_pinn import DeepEnsemblePINN
from pina._src.solver.ensemble_solver.ensemble_supervised import (
    DeepEnsembleSupervisedSolver,
)

from pina._src.solver.garom import GAROM

from pina._src.solver.autoregressive_solver import AutoregressiveSolver
