import configparser as cp
import line_profiler
import atexit
from core.dataset.ecg import BatchGenerator
from core.dataset import ECGDataset

from core.dataset.hea_loader import HeaLoader, HeaLoaderExcel

if __name__ == "__main__":
    profile = line_profiler.LineProfiler()
    atexit.register(profile.print_stats)
    mitdb_path = "G:\Team Drives\CS274C Final Project\Dataset\mitdb"
    heaLoader_mit = HeaLoader.load(mitdb_path, "mitdb_labeled.xlsx")
    train_set = ECGDataset.from_directory(mitdb_path, heaLoader_mit)

    config = cp.ConfigParser()
    config.read("config.ini")
    train_set = train_set[0]
    print(train_set)
    train_generator = BatchGenerator(train_set, config, enable_augmentation=False, logger="train_sequencer")
    profile.add_function(HeaLoaderExcel.get_label)
    profile.enable_by_count()
    print(train_generator.dump_labels())
    # print(dev_generator.dump_labels())
    # sv = SequenceVisualizer(config, experiment_env)
    # sv.visualize(train_generator, batch_limit=None, segment_limit=15)
    # sv.visualize(dev_generator, batch_limit=None, segment_limit=15)
