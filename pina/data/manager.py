"""Module for condition data management.

:Example:

    >>> from pina.data.manager import _DataManager, _GraphDataManager
    >>> # Internal API used by DataModule for batch management
"""

__all__ = [
    "_BatchManager",
    "_DataManagerInterface",
    "_DataManager",
    "_TensorDataManager",
    "_GraphDataManager",
]

from pina._src.data.manager.batch_manager import _BatchManager
from pina._src.data.manager.data_manager import _DataManager
from pina._src.data.manager.tensor_data_manager import _TensorDataManager
from pina._src.data.manager.graph_data_manager import _GraphDataManager
from pina._src.data.manager.data_manager_interface import _DataManagerInterface
