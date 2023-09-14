""" Module for AbstractProblem class """
from abc import ABCMeta, abstractmethod
from ..utils import merge_tensors, check_consistency
import torch


class AbstractProblem(metaclass=ABCMeta):
    """
    The abstract `AbstractProblem` class. All the class defining a PINA Problem
    should be inheritied from this class.

    In the definition of a PINA problem, the fundamental elements are:
    the output variables, the condition(s), and the domain(s) where the
    conditions are applied.
    """

    def __init__(self):

        # variable storing all points
        self.input_pts = {}

        # varible to check if sampling is done. If no location
        # element is presented in Condition this variable is set to true
        self._have_sampled_points = {}

        # put in self.input_pts all the points that we don't need to sample
        self._span_condition_points()
      
    @property
    def input_variables(self):
        """
        The input variables of the AbstractProblem, whose type depends on the
        type of domain (spatial, temporal, and parameter).

        :return: the input variables of self
        :rtype: list

        """
        variables = []

        if hasattr(self, 'spatial_variables'):
            variables += self.spatial_variables
        if hasattr(self, 'temporal_variable'):
            variables += self.temporal_variable
        if hasattr(self, 'parameters'):
            variables += self.parameters
        if hasattr(self, 'custom_variables'):
            variables += self.custom_variables

        return variables

    @property
    def domain(self):
        """
        The domain(s) where the conditions of the AbstractProblem are valid.

        :return: the domain(s) of self
        :rtype: list (if more than one domain are defined),
            `Span` domain (of only one domain is defined)
        """
        domains = [
            getattr(self, f'{t}_domain')
            for t in ['spatial', 'temporal', 'parameter']
            if hasattr(self, f'{t}_domain')
        ]

        if len(domains) == 1:
            return domains[0]
        elif len(domains) == 0:
            raise RuntimeError

        if len(set(map(type, domains))) == 1:
            domain = domains[0].__class__({})
            [domain.update(d) for d in domains]
            return domain
        else:
            raise RuntimeError('different domains')

    @input_variables.setter
    def input_variables(self, variables):
        raise RuntimeError

    @property
    @abstractmethod
    def output_variables(self):
        """
        The output variables of the problem.
        """
        pass

    @property
    @abstractmethod
    def conditions(self):
        """
        The conditions of the problem.
        """
        pass

    def _span_condition_points(self):
        """
        Simple function to get the condition points
        """
        for condition_name in self.conditions:
            condition = self.conditions[condition_name]
            if hasattr(condition, 'equation') and hasattr(condition, 'input_points'):
                samples = condition.input_points
            elif hasattr(condition, 'output_points') and hasattr(condition, 'input_points'):
                samples = (condition.input_points, condition.output_points)
            # skip if we need to sample
            elif hasattr(condition, 'location'):
                self._have_sampled_points[condition_name] = False
                continue
            self.input_pts[condition_name] = samples

    def discretise_domain(self, n, mode = 'random', variables = 'all', locations = 'all'):
        """
        Generate a set of points to span the `Location` of all the conditions of
        the problem.

        :param n: Number of points to sample, see Note below
            for reference.
        :type n: int
        :param mode: Mode for sampling, defaults to ``random``.
            Available modes include: random sampling, ``random``;
            latin hypercube sampling, ``latin`` or ``lh``;
            chebyshev sampling, ``chebyshev``; grid sampling ``grid``.
        :param variables: problem's variables to be sampled, defaults to 'all'.
        :type variables: str or list[str], optional
        :param locations: problem's locations from where to sample, defaults to 'all'.
        :type locations: str, optional

        :Example:
            >>> pinn.discretise_domain(n=10, mode='grid')
            >>> pinn.discretise_domain(n=10, mode='grid', location=['bound1'])
            >>> pinn.discretise_domain(n=10, mode='grid', variables=['x'])

        .. warning::
            ``random`` is currently the only implemented ``mode`` for all geometries, i.e.
            ``EllipsoidDomain``, ``CartesianDomain``, ``SimplexDomain`` and the geometries
            compositions ``Union``, ``Difference``, ``Exclusion``, ``Intersection``. The
            modes ``latin`` or ``lh``,  ``chebyshev``, ``grid`` are only implemented for
            ``CartesianDomain``.
        """

        # check consistecy n
        check_consistency(n, int)

        # check consistency mode
        check_consistency(mode, str)
        if mode not in ['random', 'grid', 'lh', 'chebyshev', 'latin']:
            raise TypeError(f'mode {mode} not valid.')
        
        # check consistency variables
        if variables == 'all':
            variables = self.input_variables
        else:
            check_consistency(variables, str)
        
        if sorted(variables) !=  sorted(self.input_variables):
            TypeError(f'Wrong variables for sampling. Variables ',
                      f'should be in {self.input_variables}.')
            
        # check consistency location
        if locations == 'all':
            locations = [condition for condition in self.conditions]
        else:
            check_consistency(locations, str)

        if sorted(locations) !=  sorted(self.conditions):
            TypeError(f'Wrong locations for sampling. Location ',
                      f'should be in {self.conditions}.')

        # sampling
        for location in locations:
            condition = self.conditions[location]

            # we try to check if we have already sampled
            try:
                already_sampled = [self.input_pts[location]]
            # if we have not sampled, a key error is thrown 
            except KeyError:
                already_sampled = []

            # if we have already sampled fully the condition
            # but we want to sample again we set already_sampled
            # to an empty list since we need to sample again, and
            # self._have_sampled_points to False.
            if self._have_sampled_points[location]:
                already_sampled = []
                self._have_sampled_points[location] = False

            # build samples
            samples = [condition.location.sample(
                            n=n,
                            mode=mode,
                            variables=variables)
                        ] + already_sampled
            pts = merge_tensors(samples)
            self.input_pts[location] = pts

            # the condition is sampled if input_pts contains all labels
            if sorted(self.input_pts[location].labels) ==  sorted(self.input_variables): 
                self._have_sampled_points[location] = True

    def add_points(self, new_points):
        """
        Adding points to the already sampled points

        :param dict new_points: a dictionary with key the location to add the points
            and values the torch.Tensor points.
        """

        if sorted(new_points.keys()) !=  sorted(self.conditions):
            TypeError(f'Wrong locations for new points. Location ',
                      f'should be in {self.conditions}.')
            
        for location in new_points.keys():
            # extract old and new points
            old_pts = self.input_pts[location]
            new_pts = new_points[location]

            # if they don't have the same variables error
            if sorted(old_pts.labels) !=  sorted(new_pts.labels):
                TypeError(f'Not matching variables for old and new points '
                          f'in condition {location}.')
            if old_pts.labels != new_pts.labels:
                new_pts = torch.hstack([new_pts.extract([i]) for i in old_pts.labels])
                new_pts.labels = old_pts.labels

            # merging
            merged_pts = torch.vstack([old_pts, new_points[location]])
            merged_pts.labels = old_pts.labels
            self.input_pts[location] = merged_pts

    @property
    def have_sampled_points(self):
        """
        Check if all points for 
        ``'Location'`` are sampled.
        """ 
        return all(self._have_sampled_points.values())
    
    @property
    def not_sampled_points(self):
        """Check which points are 
        not sampled.
        """
        # variables which are not sampled
        not_sampled = None
        if self.have_sampled_points is False:
            # check which one are not sampled:
            not_sampled = []
            for condition_name, is_sample in self._have_sampled_points.items():
                if not is_sample:
                    not_sampled.append(condition_name)
        return not_sampled
            
