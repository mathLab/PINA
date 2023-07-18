
import lightning.pytorch as pl


class PinaLogger(object):  # pl.Trainer):

    def __init__(self, loggers) -> None:

        self.loggers = loggers

        if len(self.loggers) > 1:
            raise NotImplementedError(
                'Currently implementing one logger at a time.')

    def pina_log_graph(self, model):
        self.loggers[0].log_graph(model=model)
