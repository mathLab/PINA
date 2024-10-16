class Batch:
    def __init__(self, idx_dict, dataset_dict) -> None:
        """
        TODO
        """
        self.coordinates_dict = idx_dict
        self.dataset_dict = dataset_dict

    def __len__(self):
        length = 0
        for k,v in self.coordinates_dict.items():
            length += len(v)
        return length

    def __getitem__(self, item):
        if isinstance(item, str):
            item = [item]
        if len(item) == 1:
            return self.dataset_dict[item[0]][list(self.coordinates_dict[item[0]])]
        else:
            return self.dataset_dict[item[0]][item[1]][list(self.coordinates_dict[item[0]])]
