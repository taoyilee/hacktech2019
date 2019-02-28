from core import Action
from core.dataset.ecg import BatchGenerator
from core.dataset import ECGDataset
from core.dataset.hea_loader import HeaLoader
from core.util.logger import LoggerFactory


class Preprocessor(Action):
    mitdb = None
    nsrdb = None

    def __init__(self, config, experiment_env):
        super(Preprocessor, self).__init__(config, experiment_env)
        self.init_ecg_datasets()

    def init_ecg_datasets(self, logger=None):
        mitdb_path, nsrdb_path = self.config["mitdb"].get("dataset_path"), self.config["nsrdb"].get("dataset_path")
        if logger is None:
            heaLoader_mit = HeaLoader.load(self.config, mitdb_path, self.config["mitdb"].get("excel_label"))
            heaLoader_nsr = HeaLoader.load(self.config, nsrdb_path, self.config["preprocessing"].getint("NSR_DB_TAG"))
        else:
            heaLoader_mit = HeaLoader.load(self.config, mitdb_path, self.config["mitdb"].get("excel_label"),
                                           logger=LoggerFactory(self.config).get_logger(logger_name="mit_loader"))
            heaLoader_nsr = HeaLoader.load(self.config, nsrdb_path, self.config["preprocessing"].getint("NSR_DB_TAG"),
                                           logger=LoggerFactory(self.config).get_logger(logger_name="nsr_loader"))
        self.mitdb = ECGDataset.from_directory(mitdb_path, heaLoader_mit)
        self.nsrdb = ECGDataset.from_directory(nsrdb_path, heaLoader_nsr)

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
        nsrdb_slice = nsrdb[dev_slice]
        for ticket in nsrdb_slice.tickets:
          ticket.max_index = int(ticket.siglen*.26) # .26 is how much of the signal we want to keep so we don't have an imbalanced dev set

        dev_set = mitdb[dev_slice] + nsrdb[dev_slice]  # type: ECGDataset
        dev_set.name = "development_set"
        test_set = mitdb[test_slice] + nsrdb[test_slice]  # type: ECGDataset
        test_set.name = "test_set"
        train_set = mitdb[train_slice] + nsrdb[train_slice]  # type: ECGDataset
        train_set.name = "training_set"
        return train_set, dev_set, test_set

    def preprocess(self):
        train_set, dev_set, test_set = self.split_dataset(self.mitdb, self.nsrdb)
        self.experiment_env.add_key(
            **{d.name: d.save(self.experiment_env.output_dir) for d in [train_set, dev_set, test_set]})
        train_generator = BatchGenerator(train_set, self.config, enable_augmentation=True, logger="train_sequencer")
        dev_generator = BatchGenerator(dev_set, self.config, enable_augmentation=False, logger="dev_sequencer")
        return train_generator, dev_generator