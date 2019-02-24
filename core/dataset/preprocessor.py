from core import Action
from core.dataset.ecg import BatchGenerator
from core.dataset import ECGDataset


class Preprocessor(Action):
    def __init__(self, config, experiment_env):
        super(Preprocessor, self).__init__(config, experiment_env)

    def split_dataset(self, mitdb: ECGDataset, nsrdb: ECGDataset):
        mitdb.shuffle()
        nsrdb.shuffle()
        mixture_db = mitdb + nsrdb
        mixture_db.name = "mixture_db"

        dev_record_each = self.config["preprocessing"].getint("dev_record_each")
        test_record_each = self.config["preprocessing"].getint("test_record_each")
        dev_slice = slice(None, dev_record_each)
        test_slice = slice(dev_record_each, dev_record_each + test_record_each)
        train_slice = slice(dev_record_each + test_record_each, None)
        dev_set = mitdb[dev_slice] + nsrdb[dev_slice]  # type: ECGDataset
        dev_set.name = "development_set"
        test_set = mitdb[test_slice] + nsrdb[test_slice]  # type: ECGDataset
        test_set.name = "test_set"
        train_set = mitdb[train_slice] + nsrdb[train_slice]  # type: ECGDataset
        train_set.name = "training_set"
        return train_set, dev_set, test_set

    def preprocess(self, mitdb: ECGDataset, nsrdb: ECGDataset):
        train_set, dev_set, test_set = self.split_dataset(mitdb, nsrdb)
        for data_set in [train_set, dev_set, test_set]:
            print(data_set)
            data_set.save(self.experiment_env.output_dir)
        train_generator = BatchGenerator(train_set, self.config, enable_augmentation=True, logger="train_sequencer")
        dev_generator = BatchGenerator(dev_set, self.config, enable_augmentation=False, logger="dev_sequencer")
        return train_generator, dev_generator
