import torch
from pina import LabelTensor
from pina.graph import Graph
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from pina.graph import LabelBatch
from pina.equation.equation_interface import EquationInterface
from abc import ABC, abstractmethod


class _BatchManager:
    def __init__(self, **dict):
        self.keys = list(dict.keys())
        for k, v in dict.items():
            setattr(self, k, v)

    def to(self, device):
        for k in self.keys:
            val = getattr(self, k)
            setattr(self, k, val.to(device))
        return self


class _DataManager(ABC):
    """Interfaccia base ottimizzata per la gestione dei dati."""

    def __new__(cls, **kwargs):
        # Dispatching Factory
        if cls is not _DataManager:
            return super().__new__(cls)

        # Determina se usare il gestore Tensori o Grafi
        # (Controllo ottimizzato: evita cicli se possibile)
        is_tensor_only = all(
            isinstance(v, (torch.Tensor, LabelTensor, EquationInterface))
            for v in kwargs.values()
        )

        subclass = _TensorDataManager if is_tensor_only else _GraphDataManager
        return super().__new__(subclass)

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def to_dict(self):
        return {k: getattr(self, k) for k in self.keys}


# --- GESTORE TENSORI ---


class _TensorDataManager(_DataManager):
    def __init__(self, **kwargs):
        self.keys = list(kwargs.keys())
        self._data = kwargs  # Memorizzazione in dizionario per accesso O(1)

        # # Identifica i tensori una sola volta
        # self._tensor_keys = [
        #     k for k, v in kwargs.items()
        #     if isinstance(v, (torch.Tensor, LabelTensor))
        # ]

        # Espone le chiavi come attributi (facoltativo, ma mantiene compatibilità)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __len__(self) -> int:
        # Prende la lunghezza dal primo tensore disponibile
        return self._data[self.keys[0]].shape[0]

    def __getitem__(self, idx):
        # Mapping efficiente degli elementi
        new_data = {
            k: (self._data[k][idx] if k in self.keys else self._data[k])
            for k in self.keys
        }
        return _TensorDataManager(**new_data)

    @staticmethod
    def _create_batch(items):
        if not items:
            return None
        first = items[0]
        batch_data = {}

        for k in first.keys:
            vals = [it._data[k] for it in items]
            sample = vals[0]

            if isinstance(sample, (torch.Tensor, LabelTensor)):
                batch_fn = (
                    LabelTensor.stack
                    if isinstance(sample, LabelTensor)
                    else torch.stack
                )
                batch_data[k] = batch_fn(vals, dim=0)
            else:
                batch_data[k] = sample

        return _BatchManager(**batch_data)


class _GraphDataManager(_DataManager):
    def __init__(self, **kwargs):
        self.keys = list(kwargs.keys())

        self.graph_key = next(
            k
            for k, v in kwargs.items()
            if isinstance(v, (Graph, Data, list, tuple))
        )

        self.keys = [
            k
            for k in self.keys
            if k != self.graph_key
            and isinstance(kwargs[k], (torch.Tensor, LabelTensor))
        ]

        # Prepara la lista di grafi internamente
        self.data = self._prepare_graphs(kwargs)

    def _prepare_graphs(self, kwargs):
        graphs = kwargs[self.graph_key]
        if not isinstance(graphs, (list, tuple)):
            graphs = [graphs]

        # Iniezione attributi nei grafi
        for k in self.keys:
            val_source = kwargs[k]
            # Ottimizzazione: se la lunghezza coincide, distribuiamo i tensori,
            # altrimenti trattiamo il tensore come costante per tutti.
            use_idx = (
                len(val_source) == len(graphs)
                if hasattr(val_source, "__len__")
                else False
            )

            for i, g in enumerate(graphs):
                setattr(g, k, val_source[i] if use_idx else val_source)
        return graphs

    def __len__(self) -> int:
        return len(self.data)

    def __getattr__(self, name):

        # If the requested attribute is a tensor key, stack the tensors from
        # all graphs
        if name in self.keys:
            tensors = [getattr(g, name) for g in self.data]
            batch_fn = (
                LabelTensor.stack
                if isinstance(tensors[0], LabelTensor)
                else torch.stack
            )
            return batch_fn(tensors)

        # If the requested attribute is the graph key, return the graphs
        if name == self.graph_key:
            return self.data if len(self.data) > 1 else self.data[0]

        super().__getattribute__(name)

    @classmethod
    def _init_from_graphs_list(cls, graphs, graph_key, keys):
        # Create a new instance without calling __init__
        obj = _GraphDataManager.__new__(_GraphDataManager)
        obj.graph_key = graph_key
        obj.keys = keys
        # obj._tensor_keys = tensor_keys
        obj.data = graphs
        return obj

    def __getitem__(self, idx):
        # Manage int and slice directly
        if isinstance(idx, (int, slice)):
            selected = self.data[idx]
        # Manage list or tensor of indices
        elif isinstance(idx, (list, torch.Tensor)):
            selected = [self.data[i] for i in idx]
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

        # Ensure selected is a list
        if not isinstance(selected, list):
            selected = [selected]

        # Return a new _GraphDataManager instance with the selected graphs
        return _GraphDataManager._init_from_graphs_list(
            selected,
            # tensor_keys=self._tensor_keys,
            graph_key=self.graph_key,
            keys=self.keys,
        )

    def _create_batch(items):
        if not items:
            return None
        first = items[0]
        batching_fn = (
            LabelBatch.from_data_list
            if isinstance(first.data[0], Graph)
            else Batch.from_data_list
        )

        graphs_to_batch = [item.data[0] for item in items]
        batch_graph = batching_fn(graphs_to_batch)

        batch_data = {first.graph_key: batch_graph}

        for k in first.keys:
            if k == first.graph_key:
                continue
            batch_data[k] = getattr(batch_graph, k)
            delattr(batch_graph, k)
        return _BatchManager(**batch_data)
