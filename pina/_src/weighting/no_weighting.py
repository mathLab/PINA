from pina._src.weighting.scalar_weighting import ScalarWeighting


class _NoWeighting(ScalarWeighting):
    """
    Weighting strategy that leaves all loss terms unchanged.

    This is a special case of scalar weighting where a unit weight is assigned
    to every loss term, resulting in no reweighting.

    :Example:

        >>> import torch
        >>> from pina.weighting import ScalarWeighting
        >>> # Equivalent to no weighting with unit weights:
        >>> weighting = ScalarWeighting(weights=1.0)
        >>> losses = {"loss": torch.tensor(0.5)}
        >>> weighting.aggregate(losses)
        tensor(0.5000)
    """

    def __init__(self):
        """
        Initialization of the :class:`_NoWeighting` class.
        """
        super().__init__(weights=1)
