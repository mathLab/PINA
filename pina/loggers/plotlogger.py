from pina.pina_logger import PINALogger


class PlotLogger(PINALogger):
    """ Base class for all plot loggers """

    def call(self, model):
        model.log()