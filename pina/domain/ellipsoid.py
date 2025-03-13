"""Module for the Ellipsoid Domain."""

import torch
from .domain_interface import DomainInterface
from ..label_tensor import LabelTensor
from ..utils import check_consistency


class EllipsoidDomain(DomainInterface):
    """
    Implementation of the ellipsoid domain.
    """

    def __init__(self, ellipsoid_dict, sample_surface=False):
        """
        Initialization of the :class:`EllipsoidDomain` class.

        :param dict ellipsoid_dict: A dictionary where the keys are the variable
            names and the values are the domain extrema.
        :param bool sample_surface: A flag to choose the sampling strategy.
            If ``True``, samples are taken from the surface of the ellipsoid.
            If ``False``, samples are taken from the interior of the ellipsoid.
            Default is ``False``.
        :raises TypeError: If the input dictionary is not correctly formatted.

        .. warning::
            Sampling for dimensions greater or equal to 10 could result in a
            shrinkage of the ellipsoid, which degrades the quality of the
            samples. For dimensions higher than 10, see the following reference.

        .. seealso::
            **Original reference**: Dezert, Jean, and Musso, Christian.
            *An efficient method for generating points uniformly distributed
            in hyperellipsoids.*
            Proceedings of the Workshop on Estimation, Tracking and Fusion:
            A Tribute to Yaakov Bar-Shalom. 2001.

        :Example:
            >>> spatial_domain = Ellipsoid({'x':[-1, 1], 'y':[-1,1]})
        """
        self.fixed_ = {}
        self.range_ = {}
        self._centers = None
        self._axis = None

        # checking consistency
        check_consistency(sample_surface, bool)
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
    def sample_modes(self):
        """
        List of available sampling modes.

        :return: List of available sampling modes.
        :rtype: list[str]
        """
        return ["random"]

    @property
    def variables(self):
        """
        List of variables of the domain.

        :return: List of variables of the domain.
        :rtype: list[str]
        """
        return sorted(list(self.fixed_.keys()) + list(self.range_.keys()))

    def is_inside(self, point, check_border=False):
        """
        Check if a point is inside the ellipsoid.

        :param LabelTensor point: Point to be checked.
        :param bool check_border: If ``True``, the border is considered inside
            the ellipsoid. Default is ``False``.
        :raises ValueError: If the labels of the point are different from those
            passed in the ``__init__`` method.
        :return: ``True`` if the point is inside the domain,
            ``False`` otherwise.
        :rtype: bool

        .. note::
            When ``sample_surface=True`` in the ``__init__`` method, this method
            checks only those points lying on the surface of the ellipsoid.
        """
        # small check that point is labeltensor
        check_consistency(point, LabelTensor)

        # get axis ellipse as tensors
        list_dict_vals = list(self._axis.values())
        tmp = torch.tensor(list_dict_vals, dtype=torch.float)
        ax_sq = LabelTensor(tmp.reshape(1, -1) ** 2, self.variables)

        # get centers ellipse as tensors
        list_dict_vals = list(self._centers.values())
        tmp = torch.tensor(list_dict_vals, dtype=torch.float)
        centers = LabelTensor(tmp.reshape(1, -1), self.variables)

        if not all(i in ax_sq.labels for i in point.labels):
            raise ValueError(
                "point labels different from constructor"
                f" dictionary labels. Got {point.labels},"
                f" expected {ax_sq.labels}."
            )

        # point square + shift center
        point_sq = (point - centers).pow(2)
        point_sq.labels = point.labels

        # calculate ellispoid equation
        eqn = torch.sum(point_sq.extract(ax_sq.labels) / ax_sq) - 1.0

        # if we have sampled only the surface, we check that the
        # point is inside the surface border only
        if self._sample_surface:
            return torch.allclose(eqn, torch.zeros_like(eqn))

        # otherwise we check the ellipse
        if check_border:
            return bool(eqn <= 0)

        return bool(eqn < 0)

    def _sample_range(self, n, mode, variables):
        """
        Rescale the samples to fit within the specified bounds.

        :param int n: Number of points to sample.
        :param str mode: Sampling method. Default is ``random``.
        :param list[str] variables: variables whose samples must be rescaled.
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
        pairs_center = [
            (k, v) for k, v in self._centers.items() if k in variables
        ]
        _, values_center = map(list, zip(*pairs_center))
        values_center = torch.tensor(values_center)

        # get values axis
        pairs_axis = [(k, v) for k, v in self._axis.items() if k in variables]
        _, values_axis = map(list, zip(*pairs_axis))
        values_axis = torch.tensor(values_axis)

        # Sample in the unit sphere
        if mode == "random":
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

    def sample(self, n, mode="random", variables="all"):
        """
        Sampling routine.

        :param int n: Number of points to sample.
        :param str mode: Sampling method. Default is ``random``.
            Available modes: random sampling, ``random``.
        :param list[str] variables: variables to be sampled. Default is ``all``.
        :raises NotImplementedError: If the sampling mode is not implemented.
        :return: Sampled points.
        :rtype: LabelTensor

        :Example:
            >>> ellipsoid = Ellipsoid({'x':[1, 0], 'y':1})
            >>> ellipsoid.sample(n=6)
                tensor([[0.4872, 1.0000],
                        [0.2977, 1.0000],
                        [0.0422, 1.0000],
                        [0.6431, 1.0000],
                        [0.7272, 1.0000],
                        [0.8326, 1.0000]])
        """

        def _Nd_sampler(n, mode, variables):
            """
            Sample all variables together.

            :param int n: Number of points to sample.
            :param str mode: Sampling method.
            :param list[str] variables: variables to be sampled.
            :return: Sampled points.
            :rtype: list[LabelTensor]
            """
            pairs = [(k, v) for k, v in self.range_.items() if k in variables]
            keys, _ = map(list, zip(*pairs))

            result = self._sample_range(n, mode, keys)
            result = result.as_subclass(LabelTensor)
            result.labels = keys

            for variable in variables:
                if variable in self.fixed_:
                    value = self.fixed_[variable]
                    pts_variable = torch.tensor([[value]]).repeat(
                        result.shape[0], 1
                    )
                    pts_variable = pts_variable.as_subclass(LabelTensor)
                    pts_variable.labels = [variable]

                    result = result.append(pts_variable, mode="std")
            return result

        def _single_points_sample(n, variables):
            """
            Sample a single point in one dimension.

            :param int n: Number of points to sample.
            :param list[str] variables: variables to be sampled.
            :return: Sampled points.
            :rtype: list[torch.Tensor]
            """
            tmp = []
            for variable in variables:
                if variable in self.fixed_:
                    value = self.fixed_[variable]
                    pts_variable = torch.tensor([[value]]).repeat(n, 1)
                    pts_variable = pts_variable.as_subclass(LabelTensor)
                    pts_variable.labels = [variable]
                    tmp.append(pts_variable)

            result = tmp[0]
            for i in tmp[1:]:
                result = result.append(i, mode="std")

            return result

        if variables == "all":
            variables = self.variables
        elif isinstance(variables, (list, tuple)):
            variables = sorted(variables)

        if self.fixed_ and (not self.range_):
            return _single_points_sample(n, variables).extract(variables)

        if mode in self.sample_modes:
            return _Nd_sampler(n, mode, variables).extract(variables)

        raise NotImplementedError(f"mode={mode} is not implemented.")
