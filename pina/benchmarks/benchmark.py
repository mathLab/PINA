import logging
import warnings

from abc import ABCMeta, abstractmethod
from pina.solvers import SolverInterface
from pina.trainer import Trainer

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


# TODO check problem is from BenchmarkProblemInterface otherwise some functions are not available
class Benchmark(metaclass=ABCMeta):

    def __init__(self, solvers):
        """
        Benchmark class in PINA. All the benchmarks in PINA must
        inherit from this class. A benchmark is an object which encapsulate
        PINA problems, providing mathematical description, data to assess
        if a model is correctly trained, statistics of the trained model and
        much more.

        :param solvers: The solvers to be benchmarked. It expect a single PINA
            solver instance, a list or tuple of PINA solvers, or a dict of
            PINA solvers where the keys represent the solver name.
            If a dict is passed each benchmarked solver will be indexed with the
            given key, otherwise PINA will create names following the
            convention ``solver_{idx}`` where ``idx`` is the number of the
            solver in the iterable (``solver`` if only one solver is
            passed.)
        :type solvers: dict | list | tuple | SolverInterface
        """
        # initial checks for solvers
        if isinstance(solvers, SolverInterface): # one single solver is passed
            solvers = {'solver' : solvers}
        elif isinstance(solvers, list):
            for solver in solvers:
                if not isinstance(solver, SolverInterface):
                    raise ValueError('One of the passed solver does not '
                                     'inherit from SolverInterface.')
            names = [f'solver_{idx}' for idx in range()]
            solvers = dict(zip(names, solvers))
        elif isinstance(solvers, dict):
            for solver in solvers.values():
                if not isinstance(solver, SolverInterface):
                    raise ValueError('One of the passed solver does not '
                                     'inherit from SolverInterface.')
        else:
            raise RuntimeError('Check solvers available types. Expected one '
                               'of dict | list | tuple | SolverInterface, but '
                               f'got {type(solvers)}.')
        # assign solvers
        self._solvers = solvers
        # is the benchmark completed
        self._benchmarkcompleted = False

    @staticmethod
    def benchmark_not_completed_warning(function):
        """
        Raises an error if the benchmark is not completed.
        The benchmark is completed after :meth:`start` returns.
        """
        def inner(self, *args, **kwargs):
            if not self._benchmarkcompleted:
                raise warnings.warn(
                    'The benchmark is not completed but you are '
                    'trying to compute statistics. The results '
                    'are not reliable, please run '
                    'Benchmark.start() before computing statistics.'
                    , RuntimeWarning)
            return function(self, *args, **kwargs)
        return inner

    def start(self, **kwargs):
        logging.info('PINA benchmark starting')
        # start the benchmark
        for solver_name, solver in self.solvers.items():
            print()
            logging.info(f'Solver {solver_name} benchmark')
            kwargs['default_root_dir'] = f'{type(self).__name__}/{solver_name}'
            trainer = Trainer(solver, **kwargs)
            trainer.train()
        # benchmark completed
        print()
        logging.info(f'PINA benchmark completed')
        self._benchmarkcompleted = True
        return self._benchmarkcompleted
     
    @property
    def solvers(self):
        """
        The solvers to be benchmarked.
        """
        return self._solvers
    
    # # TODO
    # @property
    # @benchmark_not_completed_warning
    # def compute_statistics(self):
    #     pass
    
    # # TODO
    # @property
    # @benchmark_not_completed_warning
    # def visualize(self):
    #     pass