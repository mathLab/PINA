"""Module for the Batch Manager class."""


class _BatchManager(dict):
    """
    Dict-like container for batched data with attribute-style access and
    convenience methods for device placement.
    """

    def to(self, device):
        """
        Move all compatible values in the batch to the specified device.

        :param device: The target device.
        :type device: torch.device | str
        :return: The updated batch manager.
        :rtype: _BatchManager
        """
        for key, value in self.items():
            if hasattr(value, "to"):
                moved_value = value.to(device)
                self[key] = moved_value

        return self

    def __getattribute__(self, name):
        """
        Provide attribute-style access to dictionary keys.

        :param str name: The name of the attribute to retrieve.
        :raises AttributeError: If the attribute is not found as a standard
            attribute or a dictionary key.
        :return: The value associated with the attribute name.
        :rtype: Any
        """
        # First, attempt to retrieve the attribute using the standard method.
        try:
            return super().__getattribute__(name)

        # If not found, attempt to retrieve the attribute as a dictionary key.
        except AttributeError:
            try:
                return self[name]
            except KeyError:
                raise AttributeError(
                    f"'BatchManager' object has no attribute '{name}'"
                )
