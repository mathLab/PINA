import h5py
from pina import LabelTensor
import os
import torch

def get_first_part(file_path):
    filename = os.path.basename(file_path)
    return filename.split('_')[0]

class Dataset:
    def __init__(self, file):
        # check and store file
        assert isinstance(file, str)
        self.file = file
        # get data
        self.pde, self.x, self.t, self.dt, self.dx = self.read_data()

    def _find_train_or_test_key(self):
        with h5py.File(self.file, 'r') as hdf:
            keys = list(hdf.keys())
            if 'train' in keys:
                return 'train'
            elif 'test' in keys:
                return 'test'
            elif 'valid' in keys:
                return 'valid'
            else:
                return None
            
    def read_data(self):
        # get the file
        f = h5py.File(self.file, 'r')
        mode = self._find_train_or_test_key()
        # get the keys
        keys = f[mode].keys()
        # get pde
        for key in keys:
            if key.startswith('pde_'):
                pde_key = key
                break
        pde = f[mode][pde_key].__array__()
        # scale Burgers equation, for lowering amplitudes (see https://github.com/brandstetter-johannes/LPSDA/blob/2a7c4252d912336d1e2be02ef533ef7684346032/common/utils.py#L109C19-L109C24)
        if get_first_part(self.file) == 'Burgers':
            pde = pde / 1000
        # get t, x, nt, nx (we use same discretization for all pdes)
        t = f[mode]['t'].__array__()
        x = f[mode]['x'].__array__()
        dt = f[mode]['dt'].__array__()[0]
        dx = f[mode]['dx'].__array__()[0]
        # convert to LabelTensors, float32
        pde = LabelTensor(torch.tensor(pde, dtype=torch.float).unsqueeze(-1), 'u')
        x = LabelTensor(torch.tensor(x, dtype=torch.float).unsqueeze(-1), 'x')
        t = LabelTensor(torch.tensor(t, dtype=torch.float).unsqueeze(-1), 't')
        dt = torch.tensor(dt, dtype=torch.float)
        dx = torch.tensor(dx, dtype=torch.float)
        return (pde, x, t, dt, dx)