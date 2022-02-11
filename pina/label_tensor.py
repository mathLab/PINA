import torch

class LabelTensor():

    def __init__(self, x, labels):


        if len(labels) != x.shape[1]:
            print(len(labels), x.shape[1])
            raise ValueError
        self.__labels = labels
        self.tensor = x

    def __getitem__(self, key):
        if key in self.labels:
            return self.tensor[:, self.labels.index(key)]
        else:
            return self.tensor.__getitem__(key)

    def __repr__(self):
        return self.tensor

    def __str__(self):
        return '{}\n {}\n'.format(self.labels, self.tensor)

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

