import logging

from abc import ABCMeta, abstractmethod
from pina.solvers import SolverInterface
from pina.trainer import Trainer

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

class BenchmarkInterface(metaclass=ABCMeta):

    def __init__(self, solvers):
        """
        BenchmarkInferfance class in PINA. All the benchmarks in PINA must
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
    
    @property
    @abstractmethod
    def description(self):
        """
        A brief description of the benchmark.
        """
        pass

    @property
    @abstractmethod
    def resume(self):
        """
        The resume of the considered problem. It must provide the mathematical
        formulation of the problem in LaTex format using Phython str, defining
        each input/output variable(s) and the physical domain of the problem.
        """
        pass

    @property
    @abstractmethod
    def problem_type(self):
        """
        Return a Python str with the problem type. Each problem could be either
        `physics-informed`, `data-driven` or `mixed` (if both).
        """
        pass

    @property
    @abstractmethod
    def problem(self):
        """
        Return the PINA problem.
        """
        pass

    @property
    @abstractmethod
    def data_directory(self):
        """
        Returns the data path from where to retrieve the high fiedelity solution
        data. Regardless the type of problem, all benchmarks have high fidelity
        solution data to compute statistics and perform visualization.
        """
        pass

    @property
    def start(self, *args, **kwargs):
        print(args, kwargs)
        logging.info('PINA benchmark starting')
        # start the benchmark
        for solver_name, solver in self.solvers.items():
            logging.info(f'Solver {solver_name} benchmark')
            kwargs['default_root_dir'] = f'{type(self).__name__}/{solver_name}'
            trainer = Trainer(solver, **kwargs)
            trainer.train()
        # benchmark completed
        logging.info(f'PINA benchmark completed')
        self._benchmarkcompleted = True

    @property
    def is_benchmark_completed(self):
        """
        Return if the benchmark is completed. The benchmark is completed
        after :meth:`start` is called.
        """
        return self._benchmarkcompleted
    
    @property
    def solvers(self):
        """
        The solvers to be benchmarked.
        """
        return self._solvers
    
    # TODO
    @property
    def compute_statistics(self):
        self.is_benchmark_completed()
        raise NotImplementedError
    
    # TODO
    @property
    def visualize(self):
        self.is_benchmark_completed()
        raise NotImplementedError