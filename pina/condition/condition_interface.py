from abc import ABCMeta


class ConditionInterface(metaclass=ABCMeta):

    condition_types = ['physics', 'supervised', 'unsupervised']

    def __init__(self, *args, **kwargs):
        self._condition_type = None
        self._problem = None

    @property
    def problem(self):
        return self._problem

    @problem.setter
    def problem(self, value):
        self._problem = value

    @property
    def condition_type(self):
        return self._condition_type

    @condition_type.setter
    def condition_type(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values]
        for value in values:
            if value not in ConditionInterface.condition_types:
                raise ValueError(
                    'Unavailable type of condition, expected one of'
                    f' {ConditionInterface.condition_types}.')
        self._condition_type = values
