import lightning.pytorch as pl

class PinaLogger(object):
    """
    PINA logger wrapper
    """

    def __init__(self, loggers) -> None:
        self.loggers = loggers

        if len(self.loggers) > 1:
            raise NotImplementedError(
                'Currently implementing one logger at a time.')

    def log_graph(self, model):
        self.loggers[0].log_graph(model=model)
