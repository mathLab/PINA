import torch


class Integral(object):

    def __init__(self, param):
        """Integral class for continous convolution

        :param param: type of continuous convolution
        :type param: string
        """

        if param == 'discrete':
            self.make_integral = self.integral_param_disc
        elif param == 'continuous':
            self.make_integral = self.integral_param_cont
        else:
            raise TypeError

    def __call__(self, *args, **kwds):
        return self.make_integral(*args, **kwds)

    def _prepend_zero(self, x):
        """Create bins for performing integral

        :param x: input tensor
        :type x: torch.tensor
        :return: bins for integrals
        :rtype: torch.tensor
        """
        return torch.cat((torch.zeros(1, dtype=x.dtype, device=x.device), x))

    def integral_param_disc(self, x, y, idx):
        """Perform discretize integral
            with discrete parameters

        :param x: input vector
        :type x: torch.tensor
        :param y: input vector
        :type y: torch.tensor
        :param idx: indeces for different strides
        :type idx: list
        :return: integral
        :rtype: torch.tensor
        """
        cs_idxes = self._prepend_zero(torch.cumsum(torch.tensor(idx), 0))
        cs = self._prepend_zero(torch.cumsum(x.flatten() * y.flatten(), 0))
        return cs[cs_idxes[1:]] - cs[cs_idxes[:-1]]

    def integral_param_cont(self, x, y, idx):
        """Perform discretize integral for continuous convolution
            with continuous parameters

        :param x: input vector
        :type x: torch.tensor
        :param y: input vector
        :type y: torch.tensor
        :param idx: indeces for different strides
        :type idx: list
        :return: integral
        :rtype: torch.tensor
        """
        raise NotImplementedError
