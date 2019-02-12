import numpy as np
import glob
import configparser as cp
from shutil import copyfile
import os
import logging
from core.dataset.qtdb import load_dat, split_dataset
from core.models.rnn import get_model
from core.util.experiments import setup_experiment

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger, ModelCheckpoint

from core.dataset.ecg import ECGDataset

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
config = cp.ConfigParser()
config.read("config.ini")
try:
    os.makedirs(config["logging"].get("logdir"))
except FileExistsError:
    pass
fh = logging.FileHandler(os.path.join(config["logging"].get("logdir"), "main.log"), mode="w+")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

if __name__ == "__main__":
    configuration_file = "config.ini"
    np.random.seed(0)
    config = cp.ConfigParser()
    config.read(configuration_file)
    output_dir, tag = setup_experiment(config["DEFAULT"].get("experiments_dir"))
    copyfile(configuration_file, os.path.join(output_dir, configuration_file))
    REJECTED_TAGS = tuple(config["qtdb"].get("reject_tags").split(","))
    VALID_SEGMTS = tuple(config["qtdb"].get("valid_segments").split(","))
    CATEGORIES = tuple([int(i) for i in config["qtdb"].get("category").split(",")])

    qtdbpath = config["qtdb"].get("dataset_path")
    print(f"Using qtdb dataset from {qtdbpath}")
    perct = config["qtdb"].getfloat("training_percent")
    percv = config["qtdb"].getfloat("validation_percent")

    mitdb = ECGDataset("mitdb", 1)
    print(len(mitdb))
    nsrdb = ECGDataset("nsrdb", 0)
    print(len(nsrdb))

    mixture_db = mitdb + nsrdb
    print(len(mixture_db))
    mixture_db[0] # get a single record from our datset #<wfdb.io.record.Record object>


    training_samples = int(perct * len(mixture_db))
    dev_samples = int(percv * len(mixture_db))
    test_samples = training_samples + dev_samples

    print(mixture_db[0].get_segment(1,10))
    mixture_db.shuffle()
    print(mixture_db[0].get_segment(1,10))
    train_set = mixture_db[:training_samples]
    dev_set = mixture_db[training_samples:dev_samples]
    test_set = mixture_db[test_samples:]

    #train_generator = train_set.create_generator()
    train_generator = train_set.to_sequence_generator()

    # finish everything above by some deadline
    #model = RNN()
    #model.fit_generator(train_generator)

    """
    exclude = set()
    exclude.update(config["qtdb"].get("excluded_records").split(","))

    initial_weights = config["RNN-train"].get("initial_weights")
    model_output = config["RNN-train"].get("model_output")
    epochs = config["RNN-train"].getint("epochs")
    tagged_data = load_dat(glob.glob(qtdbpath + "*.dat"), VALID_SEGMTS, CATEGORIES, exclude, REJECTED_TAGS)
    train_set, dev_set, test_set = split_dataset(tagged_data, config["qtdb"].getint("training_percent"),
                                                 config["qtdb"].getint("validation_percent"),
                                                 config["qtdb"].getint("testing_percent"))
    train_set.save(output_dir), dev_set.save(output_dir), test_set.save(output_dir)
    generator_args = {"sequence_length": config["RNN-train"].getint("sequence_length"),
                      "overlap_percent": config["RNN-train"].getfloat("overlap_percent"),
                      "batch_size": config["RNN-train"].getint("batch_size")}
    if config["qtdb"].getboolean("augmentation"):
        logger.log(logging.INFO, f"Data augmentation enabled")
        trn_generator_args = {"sequence_length": config["RNN-train"].getint("sequence_length"),
                              "random_time_scale_percent": config["qtdb"].getfloat("random_time_scale_percent"),
                              "dilation_factor": config["qtdb"].getfloat("dilation_factor"),
                              "batch_size": config["RNN-train"].getint("batch_size"),
                              "awgn_rms_percent": config["qtdb"].getfloat("awgn_rms_percent")}
        logger.log(logging.INFO, f"generator_args = {trn_generator_args}")
        train_generator = train_set.to_sequence_generator_augmented(**trn_generator_args)
    else:
        logger.log(logging.INFO, f"Data augmentation disabled")
        logger.log(logging.INFO, f"generator_args = {generator_args}")
        train_generator = train_set.to_sequence_generator(**generator_args)

    dev_generator = dev_set.to_sequence_generator(**generator_args)
    model = get_model(config["RNN-train"].getint("sequence_length"), train_set.features, len(set(CATEGORIES)), config)

    with open(os.path.join(output_dir, "model.json"), "w") as f:
        f.write(model.to_json())

    callbacks = [ModelCheckpoint(os.path.join(output_dir, "weights.{epoch:02d}.h5"), monitor='val_loss', verbose=0,
                                 save_best_only=False, save_weights_only=False, mode='auto', period=1),
                 CSVLogger(os.path.join(output_dir, f"training.csv"), separator=',', append=False),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                   patience=config["RNN-train"].getint("patientce_reduce_lr"),
                                   verbose=config["RNN-train"].getint("verbosity"), mode='min', min_delta=1e-6,
                                   cooldown=0, min_lr=1e-12),
                 TensorBoard(log_dir=os.path.join(config["RNN-train"].get("tensorboard_dir"), tag), histogram_freq=0,
                             batch_size=32, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0,
                             embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                             update_freq='epoch')]
    if config["RNN-train"].getboolean("early_stop"):
        logger.log(logging.INFO, f"Early Stop enabled")
        callbacks.append(
            EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=5, verbose=1, mode='min',
                          baseline=None, restore_best_weights=False))
    else:
        logger.log(logging.INFO, f"Early Stop disabled")
    model.fit_generator(train_generator, steps_per_epoch=None, epochs=config["RNN-train"].getint("epochs"), verbose=1,
                        callbacks=callbacks, validation_data=dev_generator, max_queue_size=10, workers=4,
                        use_multiprocessing=False, shuffle=True)
    model.save(os.path.join(output_dir, "final_weights.h5"))
    """
