import pytest

import matplotlib.pyplot as plt
from pina.plotter import Plotter
from pina import Trainer
from pina.model import FeedForward
from pina import PINN
from pina.problem import SpatialProblem
from pina.operators import grad
from pina import Condition, CartesianDomain
from pina.equation.equation import Equation
#from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

import torch


def test_loggers():
    pass