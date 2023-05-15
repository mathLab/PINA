import torch

from .location import Location
from ..label_tensor import LabelTensor


class EllipsoidDomain(Location):
    """PINA implementation of Ellipsoid domain."""

    def __init__(self, ellipsoid_dict, sample_surface=False):
        """PINA implementation of Ellipsoid domain.

        :param ellipsoid_dict: A dictionary with dict-key a string representing
            the input variables for the pinn, and dict-value a list with
            the domain extrema.
        :type ellipsoid_dict: dict
        :param sample_surface: A variable for choosing sample strategies. If
            `sample_surface=True` only samples on the ellipsoid surface
            frontier are taken. If `sample_surface=False` only samples on
            the ellipsoid interior are taken, defaults to False.
        :type sample_surface: bool, optional

        .. warning::
            Sampling for dimensions greater or equal to 10 could result
            in a shrinking of the ellipsoid, which degrades the quality
            of the samples. For dimensions higher than 10, other algorithms
            for sampling should be used, such as: Dezert, Jean, and Christian
            Musso. "An efficient method for generating points uniformly
            distributed in hyperellipsoids." Proceedings of the Workshop on
            Estimation, Tracking and Fusion: A Tribute to Yaakov Bar-Shalom.
            Vol. 7. No. 8. 2001.

        :Example:
            >>> spatial_domain = Ellipsoid({'x':[-1, 1], 'y':[-1,1]})

        """
        self.fixed_ = {}
        self.range_ = {}
        self._centers = None
        self._axis = None

        if not isinstance(sample_surface, bool):
            raise ValueError('sample_surface must be bool type.')

        self._sample_surface = sample_surface

        for k, v in ellipsoid_dict.items():
            if isinstance(v, (int, float)):
                self.fixed_[k] = v
            elif isinstance(v, (list, tuple)) and len(v) == 2:
                self.range_[k] = v
            else:
                raise TypeError

        # perform operation only for not fixed variables (if any)

        if self.range_:

            # convert dict vals to torch [dim, 2] matrix
            list_dict_vals = list(self.range_.values())
            tmp = torch.tensor(list_dict_vals, dtype=torch.float)

            # get the ellipsoid center
            normal_basis = torch.eye(len(list_dict_vals))
            centers = torch.diag(normal_basis * tmp.mean(axis=1))

            # get the ellipsoid axis
            ellipsoid_axis = (tmp - centers.reshape(-1, 1))[:, -1]

            # save elipsoid axis and centers as dict
            self._centers = dict(zip(self.range_.keys(), centers.tolist()))
            self._axis = dict(zip(self.range_.keys(), ellipsoid_axis.tolist()))

    @property
    def variables(self):
        """Spatial variables.

        :return: Spatial variables defined in '__init__()'
        :rtype: list[str]
        """
        return list(self.fixed_.keys()) + list(self.range_.keys())

    def is_inside(self, point, check_border=False):
        """Check if a point is inside the ellipsoid.

        :param point: Point to be checked
        :type point: LabelTensor
        :param check_border: Check if the point is also on the frontier
            of the ellipsoid, default False.
        :type check_border: bool
        :return: Returning True if the point is inside, False otherwise.
        :rtype: bool
        """

        if not isinstance(point, LabelTensor):
            raise ValueError('point expected to be LabelTensor.')

        # get axis ellipse
        list_dict_vals = list(self._axis.values())
        tmp = torch.tensor(list_dict_vals, dtype=torch.float)
        ax_sq = LabelTensor(tmp.reshape(1, -1)**2, list(self._axis.keys()))

        if not all([i in ax_sq.labels for i in point.labels]):
            raise ValueError('point labels different from constructor'
                             f' dictionary labels. Got {point.labels},'
                             f' expected {ax_sq.labels}.')

        # point square
        point_sq = point.pow(2)
        point_sq.labels = point.labels

        # calculate ellispoid equation
        eqn = torch.sum(point_sq.extract(ax_sq.labels) / ax_sq) - 1.

        if check_border:
            return bool(eqn <= 0)

        return bool(eqn < 0)

    def _sample_range(self, n, mode, variables):
        """Rescale the samples to the correct bounds.

        :param n: Number of points to sample in the ellipsoid.
        :type n: int
        :param mode: Mode for sampling, defaults to 'random'.
            Available modes include: random sampling, 'random'.
        :type mode: str, optional
        :param variables: Variables to  be rescaled in the samples.
        :type variables: torch.Tensor
        :return: Rescaled sample points.
        :rtype: torch.Tensor
        """

        # =============== For Developers ================ #
        #
        # The sampling startegy used is fairly simple.
        # For all `mode`s first we sample from the unit
        # sphere and then we scale and shift according
        # to self._axis.values() and self._centers.values().
        #
        # =============================================== #

        # get dimension
        dim = len(variables)

        # get values center
        pairs_center = [(k, v) for k, v in self._centers.items()
                        if k in variables]
        _, values_center = map(list, zip(*pairs_center))
        values_center = torch.tensor(values_center)

        # get values axis
        pairs_axis = [(k, v) for k, v in self._axis.items() if k in variables]
        _, values_axis = map(list, zip(*pairs_axis))
        values_axis = torch.tensor(values_axis)

        # Sample in the unit sphere
        if mode == 'random':
            # 1. Sample n points from the surface of a unit sphere
            # 2. Scale each dimension using torch.rand()
            #    (a random number between 0-1) so that it lies within
            #    the sphere, only if self._sample_surface=False
            # 3. Multiply with self._axis.values() to make it ellipsoid
            # 4. Shift the mean of the ellipse by adding self._centers.values()

            # step 1.
            pts = torch.randn(size=(n, dim))
            pts = pts / torch.linalg.norm(pts, axis=-1).view((n, 1))
            if not self._sample_surface:  # step 2.
                scale = torch.rand((n, 1))
                pts = pts * scale

            # step 3. and 4.
            pts *= values_axis
            pts += values_center

        return pts

    def sample(self, n, mode='random', variables='all'):
        """Sample routine.

        :param n: Number of points to sample in the ellipsoid.
        :type n: int
        :param mode: Mode for sampling, defaults to 'random'.
            Available modes include: random sampling, 'random'.
        :type mode: str, optional
        :param variables: pinn variable to be sampled, defaults to 'all'.
        :type variables: str or list[str], optional

        :Example:
            >>> elips = Ellipsoid({'x':[1, 0], 'y':1})
            >>> elips.sample(n=6)
                tensor([[0.4872, 1.0000],
                        [0.2977, 1.0000],
                        [0.0422, 1.0000],
                        [0.6431, 1.0000],
                        [0.7272, 1.0000],
                        [0.8326, 1.0000]])
        """

        def _Nd_sampler(n, mode, variables):
            """Sample all the variables together

            :param n: Number of points to sample.
            :type n: int
            :param mode: Mode for sampling, defaults to 'random'.
                Available modes include: random sampling, 'random';
                latin hypercube sampling, 'latin' or 'lh';
                chebyshev sampling, 'chebyshev'; grid sampling 'grid'.
            :type mode: str, optional.
            :param variables: pinn variable to be sampled, defaults to 'all'.
            :type variables: str or list[str], optional.
            :return: Sample points.
            :rtype: list[torch.Tensor]
            """
            pairs = [(k, v) for k, v in self.range_.items() if k in variables]
            keys, _ = map(list, zip(*pairs))

            result = self._sample_range(n, mode, keys)
            result = result.as_subclass(LabelTensor)
            result.labels = keys

            for variable in variables:
                if variable in self.fixed_.keys():
                    value = self.fixed_[variable]
                    pts_variable = torch.tensor([[value]
                                                 ]).repeat(result.shape[0], 1)
                    pts_variable = pts_variable.as_subclass(LabelTensor)
                    pts_variable.labels = [variable]

                    result = result.append(pts_variable, mode='std')
            return result

        def _single_points_sample(n, variables):
            """Sample a single point in one dimension.

            :param n: Number of points to sample.
            :type n: int
            :param variables: Variables to sample from.
            :type variables: list[str]
            :return: Sample points.
            :rtype: list[torch.Tensor]
            """
            tmp = []
            for variable in variables:
                if variable in self.fixed_.keys():
                    value = self.fixed_[variable]
                    pts_variable = torch.tensor([[value]]).repeat(n, 1)
                    pts_variable = pts_variable.as_subclass(LabelTensor)
                    pts_variable.labels = [variable]
                    tmp.append(pts_variable)

            result = tmp[0]
            for i in tmp[1:]:
                result = result.append(i, mode='std')

            return result

        if self.fixed_ and (not self.range_):
            return _single_points_sample(n, variables)

        if variables == 'all':
            variables = list(self.range_.keys()) + list(self.fixed_.keys())

        if mode in ['random']:
            return _Nd_sampler(n, mode, variables)
        else:
            raise ValueError(f'mode={mode} is not valid.')
