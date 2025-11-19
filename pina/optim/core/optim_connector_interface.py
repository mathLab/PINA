"""Module for the PINA Optimizer and Scheduler Connectors Interface."""

from abc import ABCMeta, abstractmethod
from functools import wraps


class OptimizerConnectorInterface(metaclass=ABCMeta):
    """
    Interface class for method definitions in the Optimizer classes.
    """

    @abstractmethod
    def parameter_hook(self, parameters):
        """
        Abstract method to define the hook logic for the optimizer. This hook
        is used to initialize the optimizer instance with the given parameters.

        :param dict parameters: The parameters of the model to be optimized.
        """

    @abstractmethod
    def solver_hook(self, solver):
        """
        Abstract method to define the hook logic for the optimizer. This hook
        is used to hook the optimizer instance with the given parameters.

        :param SolverInterface solver: The solver to hook.
        """


class SchedulerConnectorInterface(metaclass=ABCMeta):
    """
    Abstract base class for defining a scheduler. All specific schedulers should
    inherit form this class and implement the required methods.
    """

    @abstractmethod
    def optimizer_hook(self):
        """
        Abstract method to define the hook logic for the scheduler. This hook
        is used to hook the scheduler instance with the given optimizer.
        """


class _HooksOptim:
    """
    Mixin class to manage and track the execution of hook methods in optimizer
    or scheduler classes.

    This class automatically detects methods ending with `_hook` and tracks
    whether they have been executed for a given instance. Subclasses defining
    `_hook` methods benefit from automatic tracking without additional
    boilerplate.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the hooks tracking dictionary `hooks_done` for this instance.

        Each hook method detected in the class hierarchy is added to 
        `hooks_done` with an initial value of False (not executed).
        """
        super().__init__(*args, **kwargs)
        # Initialize hooks_done per instance
        self.hooks_done = {}
        for cls in self.__class__.__mro__:
            for attr_name, attr_value in cls.__dict__.items():
                if callable(attr_value) and attr_name.endswith("_hook"):
                    self.hooks_done.setdefault(attr_name, False)

    def __init_subclass__(cls, **kwargs):
        """
        Hook called when a subclass of _HooksOptim is created.

        Wraps all concrete `_hook` methods defined in the subclass so that
        executing the method automatically updates `hooks_done`.
        """
        super().__init_subclass__(**kwargs)
        # Wrap only concrete _hook methods defined in this subclass
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value) and attr_name.endswith("_hook"):
                setattr(cls, attr_name, cls.hook_wrapper(attr_name, attr_value))

    @staticmethod
    def hook_wrapper(name, func):
        """
        Wrap a hook method to mark it as executed after calling it.

        :param str name: The name of the hook method.
        :param callable func: The original hook method to wrap.
        :return: The wrapped hook method that updates `hooks_done`.
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self.hooks_done[name] = True
            return result

        return wrapper
