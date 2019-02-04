import wfdb


class Dataset:
    records = []

    def __init__(self, name):
        self.name = name

    def __len__(self):
        return len(self.records)

    @classmethod
    def from_dir(cls, dataset_dir):
        pass
