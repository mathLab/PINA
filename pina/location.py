from abc import ABCMeta, abstractmethod


class Location(metaclass=ABCMeta):

    @property
    @abstractmethod
    def sample(self):
        pass
