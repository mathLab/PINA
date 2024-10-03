
from abc import ABCMeta


class ConditionInterface(metaclass=ABCMeta):

    condition_types = ['physical', 'supervised', 'unsupervised']
    def __init__(self):
        self._condition_type = None

    @property
    def condition_type(self):
        return self._condition_type
    
    @condition_type.setattr
    def condition_type(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values]
        for value in values:
            if value not in ConditionInterface.condition_types:
                raise ValueError( 
                                'Unavailable type of condition, expected one of'
                                f' {ConditionInterface.condition_types}.'
                                )
        self._condition_type = values