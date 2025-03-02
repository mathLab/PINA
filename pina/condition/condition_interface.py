from abc import ABCMeta


class ConditionInterface(metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        self._problem = None

    @property
    def problem(self):
        return self._problem

    @problem.setter
    def problem(self, value):
        self._problem = value
