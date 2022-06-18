import torch

class LabelTensor():

    def __init__(self, x, labels):


        if len(labels) != x.shape[1]:
            raise ValueError(f"Invalid tensor shape for dim 1, expected {len(self.labels)} got {new_tensor.shape[1]}")
        self.__labels = labels
        self.__tensor = x

    def __getitem__(self, key):
        '''
        if isinstance(key, (tuple, list)):
            indeces = [self.labels.index(k) for k in key]
            return LabelTensor(self.tensor[:, indeces], [self.labels[idx] for idx in indeces])
        '''
        if isinstance(key, slice):
            indices = range(*key.indices(len(self.list)))
            return [self.tensor[i] for i in indices]
        
        if key in self.labels:
            return self.tensor[:, self.labels.index(key)]
        else:
            return self.tensor.__getitem__(key)

    def __repr__(self):
        return '{}\n'.format(self.tensor)

    def __str__(self):
        return '{}\n {}\n'.format(self.labels, self.tensor)
    
    def __mul__(self, other): 
        return LabelTensor(self.tensor*other, self.labels)
        
    def __rmul__(self, other): 
        return self.__mul__(other)
    
    def __pow__(self, power):
        return LabelTensor(torch.pow(self.tensor, power), self.labels)
    
    def _check_tensor_validity(self, new_tensor):
        if not torch.is_tensor(new_tensor):
            raise ValueError(f"Expecter Tensor object got {type(new_tensor)}")
        if len(self.labels) != new_tensor.shape[1]:
            raise ValueError(f"Invalid tensor shape for dim 1, expected {len(self.labels)} got {new_tensor.shape[1]}")

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def device(self):
        return self.tensor.device

    @property
    def labels(self):
        return self.__labels
    
    @property
    def tensor(self):
        return self.__tensor
    
    @tensor.setter
    def tensor(self, value):
        self._check_tensor_validity(value)
        self.__tensor = value
        
    @tensor.getter
    def tensor(self):
        return self.__tensor
    
    @staticmethod
    def hstack(labeltensor_list):
        concatenated_tensor = torch.cat([lt.tensor for lt in labeltensor_list], axis=1)
        concatenated_label = sum([lt.labels for lt in labeltensor_list], [])
        return LabelTensor(concatenated_tensor, concatenated_label)



if __name__ == "__main__":
    import numpy as np
    a = np.random.uniform(size=(20, 3))
    a = np.random.uniform(size=(20, 3))
    p = torch.from_numpy(a)
    t = LabelTensor(p, labels=['u', 'p', 't'])
    print(t)
    print(t['u'])
    t *= 2
    print(t['u'])
    print(t[:, 0])
    t**2
    print(t)
