"""
Module for managing batches of data with device transfer capabilities.
"""


class _BatchManager(dict):
    """
    A dictionary-based batch manager that supports dot-notation
    and moving tensors to devices.
    """

    def to(self, device):
        """
        Move all tensors in the batch to the specified device.

        :param device: The target device.
        :type device: torch.device | str
        :return: The updated batch manager.
        :rtype: _BatchManager
        """
        for key, value in self.items():
            if hasattr(value, "to"):
                moved_value = value.to(device)
                self[key] = moved_value  # Updates both dict and attribute
        return self

    def __getattribute__(self, name):
        """
        Alias attribute access to dictionary keys.

        :param str name: The name of the attribute to retrieve.
        :return: The value associated with the attribute name.
        :rtype: Any
        """
        try:
            return super().__getattribute__(name)
        except AttributeError:
            try:
                return self[name]
            except KeyError:
                raise AttributeError(
                    f"'BatchManager' object has no attribute '{name}'"
                )
